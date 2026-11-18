import re
import time
from collections.abc import Callable
from typing import Any

from huggingface_hub import HfApi, get_token

from ml_intern_mcp.compute.backend import ComputeError
from ml_intern_mcp.compute.models import Hardware, Job, JobSpec, JobState, Offer
from ml_intern_mcp.compute.store import JobStore
from ml_intern_mcp.config import HfJobsConfig

CATALOG_TTL_SECONDS = 3600
UNITS_PER_HOUR = {"second": 3600, "minute": 60, "hour": 1}
STAGES = {
    "SCHEDULING": JobState.QUEUED,
    "RUNNING": JobState.RUNNING,
    "COMPLETED": JobState.SUCCEEDED,
    "CANCELED": JobState.CANCELLED,
    "DELETED": JobState.CANCELLED,
    "ERROR": JobState.FAILED,
}


def _gigabytes(text: str | None) -> float | None:
    match = re.match(r"\s*([\d.]+)", text or "")
    return float(match.group(1)) if match else None


def offer_from_hardware_info(info: Any) -> Offer:
    accelerator = info.accelerator
    return Offer(
        hardware=info.name,
        gpu=accelerator.model if accelerator else None,
        gpu_count=int(accelerator.quantity) if accelerator else 0,
        vram_gb=_gigabytes(accelerator.vram) if accelerator else None,
        usd_per_hour=round(info.unit_cost_usd * UNITS_PER_HOUR[info.unit_label], 4),
    )


class HfJobsBackend:
    """Runs jobs on Hugging Face Jobs with `uv run`, billed per minute of the chosen flavor."""

    name = "hf_jobs"
    queues_locally = False

    def __init__(
        self,
        store: JobStore,
        config: HfJobsConfig,
        api: Any = None,
        token: Callable[[], str | None] = get_token,
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.namespace = config.namespace
        self.poll_seconds = config.poll_seconds
        self.api = api if api is not None else HfApi()
        self.token = token
        self.clock = clock
        self._offers: dict[str, Offer] = {}
        self._offers_at = float("-inf")

    def offers(self) -> dict[str, Offer]:
        if self.clock() - self._offers_at >= CATALOG_TTL_SECONDS:
            offers = [offer_from_hardware_info(i) for i in self.api.list_jobs_hardware()]
            self._offers = {offer.hardware: offer for offer in offers}
            self._offers_at = self.clock()
        return self._offers

    def hardware(self) -> Hardware:
        try:
            offers = sorted(self.offers().values(), key=lambda o: o.usd_per_hour)
        except Exception as error:  # noqa: BLE001 - any catalog failure makes the backend unavailable
            return Hardware(backend=self.name, status=f"unavailable: {error}")
        status = "ready" if self.token() else "no HF token: run 'hf auth login' or set HF_TOKEN"
        return Hardware(backend=self.name, offers=offers, status=status)

    def validate(self, spec: JobSpec) -> None:
        if spec.gpu is not None or spec.gpus != 1:
            raise ComputeError(
                "gpu and gpus apply to the local backend; pick a hardware option with the GPUs "
                "the job needs"
            )
        if not self.token():
            raise ComputeError("hf_jobs needs an HF token: run 'hf auth login' or set HF_TOKEN")
        offer = self._offer(spec)
        if spec.vram_gb is not None and offer.vram_gb is not None and spec.vram_gb > offer.vram_gb:
            raise ComputeError(
                f"the job needs {spec.vram_gb} GB of VRAM but {offer.hardware} has {offer.vram_gb} GB"
            )

    def price(self, spec: JobSpec) -> float | None:
        return self._offer(spec).usd_per_hour

    def submit(self, job: Job, gpus: list[int]) -> None:
        token = self.token()
        info = self.api.run_uv_job(
            script=str(self.store.folder(job.id) / "script.py"),
            flavor=job.spec.hardware,
            timeout=job.spec.timeout_minutes * 60,
            env={**job.spec.env, "PYTHONUNBUFFERED": "1"},
            secrets={"HF_TOKEN": token} if token else None,
            labels={"ml-intern-job": job.id},
            namespace=self.namespace,
        )
        job.remote_id = info.id
        self._apply(job, info)

    def status(self, job: Job) -> None:
        self._apply(job, self.api.inspect_job(job_id=job.remote_id, namespace=self.namespace))

    def logs(self, job: Job) -> None:
        lines = self.api.fetch_job_logs(job_id=job.remote_id, namespace=self.namespace)
        self.store.write_log(job.id, "".join(line.rstrip("\n") + "\n" for line in lines))

    def cancel(self, job: Job) -> None:
        if job.remote_id is not None:
            self.api.cancel_job(job_id=job.remote_id, namespace=self.namespace)
        job.state = JobState.CANCELLED
        job.finished_at = self.clock()

    def _offer(self, spec: JobSpec) -> Offer:
        try:
            offers = self.offers()
        except Exception as error:
            raise ComputeError(f"could not read the hf_jobs hardware catalog: {error}") from error
        if spec.hardware not in offers:
            raise ComputeError(f"hf_jobs needs hardware set to one of: {', '.join(sorted(offers))}")
        return offers[spec.hardware]

    def _apply(self, job: Job, info: Any) -> None:
        stage = getattr(info.status.stage, "value", info.status.stage) if info.status else None
        if info.started_at is not None:
            job.started_at = info.started_at.timestamp()
        job.state = STAGES.get(stage, job.state)
        if not job.state.finished:
            return
        job.finished_at = info.finished_at.timestamp() if info.finished_at else self.clock()
        if job.state == JobState.FAILED:
            timed_out = (job.duration_seconds or 0) >= job.spec.timeout_minutes * 60
            job.state = JobState.TIMED_OUT if timed_out else JobState.FAILED
            job.error = (
                info.status.message if info.status else None
            ) or "the job failed on HF Jobs"
