import base64
import json
import math
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from ml_intern_mcp.compute.backend import ComputeError
from ml_intern_mcp.compute.models import Hardware, Job, JobSpec, JobState, Offer
from ml_intern_mcp.compute.store import JobStore
from ml_intern_mcp.config import SkyPilotConfig

SKYPILOT_VERSION = "0.14.3"
BRIDGE = Path(__file__).with_name("sky_bridge.py")
CATALOG_TTL_SECONDS = 6 * 3600
UV_VERSION = "0.12.5"
KILL_GRACE_SECONDS = 30
TIMEOUT_EXIT_CODE = 124
EXIT_MARKER = "[ml-intern] exit code "
STATES = {
    "PENDING": JobState.QUEUED,
    "SUBMITTED": JobState.QUEUED,
    "STARTING": JobState.QUEUED,
    "RUNNING": JobState.RUNNING,
    "RECOVERING": JobState.RUNNING,
    "WINDING_DOWN": JobState.RUNNING,
    "CANCELLING": JobState.RUNNING,
    "SUCCEEDED": JobState.SUCCEEDED,
    "CANCELLED": JobState.CANCELLED,
}


class SkyClient(Protocol):
    def enabled_infra(self) -> list[str]: ...

    def accelerators(self, infra: list[str]) -> dict[str, list[Any]]: ...

    def launch(self, config: dict[str, Any], name: str) -> str: ...

    def job(self, name: str) -> Any | None: ...

    def request_error(self, request_id: str) -> str | None: ...

    def logs(self, name: str) -> str: ...

    def cancel(self, name: str) -> None: ...


class SkyPilotClient:
    """Calls the SkyPilot SDK through sky_bridge.py in an environment that has SkyPilot."""

    def __init__(self, infra: list[str], runner: list[str] | None = None, timeout: float = 600):
        self.runner = runner if runner is not None else bridge_runner(infra)
        self.timeout = timeout

    def call(self, command: str, *args: Any) -> Any:
        with tempfile.TemporaryDirectory(prefix="ml-intern-sky-") as folder:
            request = Path(folder) / "request.json"
            result = Path(folder) / "result.json"
            request.write_text(json.dumps({"command": command, "args": list(args)}))
            try:
                process = subprocess.run(
                    [*self.runner, str(BRIDGE), str(request), str(result)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ComputeError(f"SkyPilot {command} failed: {error}") from error
            if process.returncode != 0 or not result.exists():
                output = (process.stderr or process.stdout).strip().splitlines()[-5:]
                raise ComputeError(f"SkyPilot {command} failed: {' '.join(output)}")
            return json.loads(result.read_text())["result"]

    def enabled_infra(self) -> list[str]:
        return self.call("enabled_infra")

    def accelerators(self, infra: list[str]) -> dict[str, list[Any]]:
        clouds = sorted({entry.split("/")[0] for entry in infra})
        catalog = self.call("accelerators", clouds)
        return {
            name: [SimpleNamespace(**item) for item in items] for name, items in catalog.items()
        }

    def launch(self, config: dict[str, Any], name: str) -> str:
        return self.call("launch", config, name)

    def job(self, name: str) -> Any | None:
        record = self.call("job", name)
        return SimpleNamespace(**record) if record is not None else None

    def request_error(self, request_id: str) -> str | None:
        return self.call("request_error", request_id)

    def logs(self, name: str) -> str:
        return self.call("logs", name)

    def cancel(self, name: str) -> None:
        self.call("cancel", name)


def bridge_runner(infra: list[str]) -> list[str]:
    extras = sorted({entry.split("/")[0] for entry in infra})
    package = f"skypilot[{','.join(extras)}]" if extras else "skypilot"
    return ["uv", "run", "--no-project", "--with", f"{package}=={SKYPILOT_VERSION}", "python", "-I"]


def _hardware_name(accelerator: str, count: float) -> str:
    return f"{accelerator}:{int(count) if float(count).is_integer() else count}"


def _price(item: Any, use_spot: bool) -> float | None:
    price = item.spot_price if use_spot else item.price
    return None if price is None or math.isnan(price) else float(price)


def offers_from_catalog(catalog: dict[str, list[Any]], use_spot: bool) -> list[Offer]:
    """Keeps the cheapest instance for each accelerator and count."""
    best: dict[str, Offer] = {}
    for accelerator, items in catalog.items():
        for item in items:
            price = _price(item, use_spot)
            if price is None:
                continue
            hardware = _hardware_name(accelerator, item.accelerator_count)
            if hardware in best and best[hardware].usd_per_hour <= price:
                continue
            memory = item.device_memory
            best[hardware] = Offer(
                hardware=hardware,
                gpu=accelerator,
                gpu_count=math.ceil(item.accelerator_count),
                vram_gb=None if memory is None or math.isnan(memory) or memory < 1 else memory,
                usd_per_hour=round(price, 4),
            )
    return sorted(best.values(), key=lambda o: o.usd_per_hour)


def task_config(job: Job, script: str, config: SkyPilotConfig, hf_token: str | None) -> dict:
    """The SkyPilot task that runs the job script with uv under the job timeout."""
    timeout_seconds = job.spec.timeout_minutes * 60
    resources: dict[str, Any] = {
        "accelerators": job.spec.hardware,
        "use_spot": config.use_spot,
        "job_recovery": {
            "strategy": "EAGER_NEXT_REGION",
            "max_restarts_on_errors": config.max_restarts_on_errors,
        },
    }
    if len(config.infra) == 1:
        resources["infra"] = config.infra[0]
    elif config.infra:
        resources["any_of"] = [{"infra": infra} for infra in config.infra]
    run = "\n".join(
        [
            'export PATH="$HOME/.local/bin:$PATH"',
            "mkdir -p ~/ml-intern-job && cd ~/ml-intern-job",
            'printf %s "$ML_INTERN_SCRIPT" | base64 -d > script.py',
            (
                f"timeout --signal=TERM --kill-after={KILL_GRACE_SECONDS} {timeout_seconds} "
                "uv run --script script.py"
            ),
            "code=$?",
            f'echo "{EXIT_MARKER}$code"',
            "exit $code",
        ]
    )
    task: dict[str, Any] = {
        "name": remote_name(job),
        "resources": resources,
        "envs": {
            **job.spec.env,
            "PYTHONUNBUFFERED": "1",
            "ML_INTERN_SCRIPT": base64.b64encode(script.encode()).decode(),
        },
        "setup": f"command -v uv || curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | sh",
        "run": run,
    }
    if hf_token:
        task["secrets"] = {"HF_TOKEN": hf_token}
    return task


def remote_name(job: Job) -> str:
    return f"mli-{job.id}"


def exit_code_from_log(text: str) -> int | None:
    codes = re.findall(rf"^{re.escape(EXIT_MARKER)}(\d+)\s*$", text, flags=re.MULTILINE)
    return int(codes[-1]) if codes else None


class SkyPilotBackend:
    """Runs jobs as SkyPilot managed jobs, which relaunch on another machine after a preemption."""

    name = "skypilot"
    queues_locally = False

    def __init__(
        self,
        store: JobStore,
        config: SkyPilotConfig,
        client_factory: Callable[[], SkyClient] | None = None,
        hf_token: Callable[[], str | None] = lambda: None,
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.config = config
        self.poll_seconds = config.poll_seconds
        self.client_factory = client_factory or (lambda: SkyPilotClient(config.infra))
        self.hf_token = hf_token
        self.clock = clock
        self._client: SkyClient | None = None
        self._offers: dict[str, Offer] = {}
        self._offers_at = float("-inf")

    @property
    def client(self) -> SkyClient:
        if self._client is None:
            self._client = self.client_factory()
        return self._client

    def offers(self) -> dict[str, Offer]:
        if self.clock() - self._offers_at >= CATALOG_TTL_SECONDS:
            infra = self.config.infra or self.client.enabled_infra()
            if not infra:
                raise ComputeError("no infra is enabled: set up credentials and run 'sky check'")
            catalog = self.client.accelerators(infra)
            offers = offers_from_catalog(catalog, self.config.use_spot)
            self._offers = {offer.hardware: offer for offer in offers}
            self._offers_at = self.clock()
        return self._offers

    def hardware(self) -> Hardware:
        try:
            offers = list(self.offers().values())
        except Exception as error:  # noqa: BLE001 - any catalog failure makes the backend unavailable
            return Hardware(backend=self.name, status=f"unavailable: {error}")
        return Hardware(backend=self.name, offers=offers)

    def validate(self, spec: JobSpec) -> None:
        if spec.gpu is not None or spec.gpus != 1:
            raise ComputeError(
                "gpu and gpus apply to the local backend; pick a hardware option with the GPUs "
                "the job needs, such as A100:2"
            )
        # The catalog lists device memory per GPU on some providers and per machine on others,
        # so vram_gb is shown in list_hardware but not checked here.
        self._offer(spec)

    def price(self, spec: JobSpec) -> float | None:
        return self._offer(spec).usd_per_hour

    def submit(self, job: Job, gpus: list[int]) -> None:
        script = (self.store.folder(job.id) / "script.py").read_text()
        config = task_config(job, script, self.config, self.hf_token())
        job.remote_id = self.client.launch(config, remote_name(job))

    def status(self, job: Job) -> None:
        record = self.client.job(remote_name(job))
        if record is None:
            error = self.client.request_error(job.remote_id)
            if error is not None:
                job.state = JobState.FAILED
                job.error = f"SkyPilot could not launch the job: {error}"
                job.finished_at = self.clock()
            return
        status = str(record.status)
        if record.start_at:
            job.started_at = float(record.start_at)
        job.state = (
            JobState.FAILED if status.startswith("FAILED") else STATES.get(status, job.state)
        )
        if not job.state.finished:
            return
        job.finished_at = float(record.end_at or self.clock())
        if job.state == JobState.FAILED:
            reason = record.failure_reason
            job.error = f"{status}: {reason}" if reason else status

    def logs(self, job: Job) -> None:
        text = self.client.logs(remote_name(job))
        self.store.write_log(job.id, text)
        exit_code = exit_code_from_log(text)
        if exit_code is None or not job.state.finished:
            return
        job.exit_code = exit_code
        if exit_code == TIMEOUT_EXIT_CODE:
            job.state = JobState.TIMED_OUT

    def cancel(self, job: Job) -> None:
        if job.remote_id is not None:
            self.client.cancel(remote_name(job))
        job.state = JobState.CANCELLED
        job.finished_at = self.clock()

    def _offer(self, spec: JobSpec) -> Offer:
        try:
            offers = self.offers()
        except Exception as error:
            raise ComputeError(f"could not read the skypilot catalog: {error}") from error
        if spec.hardware not in offers:
            cheapest = ", ".join(list(offers)[:15])
            raise ComputeError(
                f"skypilot needs hardware set to an accelerator and count from list_hardware, "
                f"such as: {cheapest}"
            )
        return offers[spec.hardware]
