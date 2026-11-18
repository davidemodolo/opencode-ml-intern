from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from ml_intern_mcp.compute.backend import ComputeError
from ml_intern_mcp.compute.models import Hardware, Job, JobSpec, JobState
from ml_intern_mcp.compute.service import ComputeService

STATUS_TAIL_LINES = 20
MAX_OFFERS = 40
NonNegative = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(ge=1)]
RUNNING_NOTE = (
    "The job is still running. Its completion is notified to the session automatically: "
    "do not poll, continue with other work or end your turn."
)


def _script_source(script: str) -> str:
    candidate = script.strip()
    if "\n" in candidate or not candidate.endswith(".py"):
        return script
    path = Path(candidate)
    if not path.is_absolute():
        raise ComputeError(f"script path {candidate} must be absolute")
    if not path.is_file():
        raise ComputeError(f"script file {candidate} does not exist")
    return path.read_text()


def _tail(lines: list[str], count: int) -> list[str]:
    return lines[len(lines) - count :] if count else []


def _hardware_view(hardware: Hardware, gpu: str | None) -> dict[str, Any]:
    offers = hardware.offers
    if gpu:
        needle = gpu.lower()
        offers = [o for o in offers if needle in f"{o.hardware} {o.gpu or ''}".lower()]
    view = {**asdict(hardware), "offers": [asdict(o) for o in offers[:MAX_OFFERS]]}
    if len(offers) > MAX_OFFERS:
        view["more_offers"] = len(offers) - MAX_OFFERS
    return view


def _queued_note(service: ComputeService, job: Job) -> str:
    backend = service.backends.get(job.spec.backend)
    if backend is not None and not backend.queues_locally:
        return f"Waiting for {job.spec.backend} to start the job."
    gpus = "a free GPU" if job.spec.gpus == 1 else f"{job.spec.gpus} free GPUs"
    vram = f" with {job.spec.vram_gb} GB of free VRAM" if job.spec.vram_gb else ""
    return f"Waiting for {gpus}{vram}."


def _job_view(
    service: ComputeService, job: Job, log_lines: list[str], tail: int = STATUS_TAIL_LINES
) -> dict[str, Any]:
    view: dict[str, Any] = {
        "job_id": job.id,
        "name": job.spec.name,
        "backend": job.spec.backend,
        "hardware": job.spec.hardware,
        "state": job.state.value,
        "gpus": job.gpus,
        "exit_code": job.exit_code,
        "duration_seconds": job.duration_seconds,
        "timeout_minutes": job.spec.timeout_minutes,
        "usd_per_hour": job.usd_per_hour,
        "cost_usd": job.cost_usd,
        "error": job.error,
        "workdir": str(service.store.folder(job.id)),
        "log_tail": _tail(log_lines, tail),
    }
    if job.state == JobState.RUNNING:
        view["note"] = RUNNING_NOTE
    if job.state == JobState.QUEUED:
        view["note"] = _queued_note(service, job)
    return view


def register_compute_tools(server: MCPServer, service: ComputeService) -> None:
    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ComputeError as error:
            raise ToolError(str(error)) from error

    @server.tool(
        description=(
            "List the compute backends. 'local' lists its GPUs (name, total and free VRAM in GB); "
            "remote backends list their hardware options, cheapest first, with GPU, GPU count, "
            "VRAM and price in USD per hour, and a status that says whether their credentials "
            "work. 'gpu' keeps the options whose GPU or name contains it, such as 'A100'."
        )
    )
    def list_hardware(gpu: str | None = None) -> dict[str, Any]:
        return {"backends": [_hardware_view(h, gpu) for h in service.hardware()]}

    @server.tool(
        description=(
            "Submit a Python training or evaluation script as a background job. 'script' is the "
            "script source or the absolute path of an existing .py file, which is copied at submit "
            "time. The script runs with 'uv run --script', so declare its dependencies inline with "
            "a PEP 723 block "
            "('# /// script' ... 'dependencies = [...]' ... '# ///'). On 'local' a job gets its "
            "GPUs to itself and waits in a queue until they are free; gpus sets how many it gets "
            "together (CUDA_VISIBLE_DEVICES lists them) and vram_gb, the VRAM it needs across "
            "them, makes a job that cannot fit fail fast. On a remote backend (see list_hardware) "
            "set hardware to one of its options: the job costs its hourly price while it runs and "
            "its working directory is deleted when it ends, so push outputs to the Hub. "
            "Completion is notified to the session automatically."
        )
    )
    def submit_job(
        name: str,
        script: str,
        backend: str = "local",
        hardware: str | None = None,
        gpu: int | None = None,
        gpus: Positive = 1,
        vram_gb: float | None = None,
        timeout_minutes: int = 60,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        spec = JobSpec(
            name=name,
            script=call(_script_source, script),
            backend=backend,
            hardware=hardware,
            gpu=gpu,
            gpus=gpus,
            vram_gb=vram_gb,
            timeout_minutes=timeout_minutes,
            env=env or {},
        )
        job = call(service.submit, spec)
        return _job_view(service, job, [], tail=0)

    @server.tool(
        description="Show the state, exit code, duration, cost and last log lines of a job."
    )
    def job_status(job_id: str) -> dict[str, Any]:
        service.reconcile(force={job_id})
        job = call(service.get, job_id)
        return _job_view(service, job, service.log_lines(job_id))

    @server.tool(
        description=(
            "Read the log of a job. By default returns the last 'tail' lines; "
            "set offset to read a window of 'limit' lines from that line."
        )
    )
    def job_logs(
        job_id: str,
        tail: NonNegative = 100,
        offset: NonNegative | None = None,
        limit: Positive = 200,
    ) -> dict[str, Any]:
        lines = call(service.log_lines, job_id)
        selected = lines[offset : offset + limit] if offset is not None else _tail(lines, tail)
        return {"job_id": job_id, "total_lines": len(lines), "lines": selected}

    @server.tool(description="Cancel a queued or running job.")
    def cancel_job(job_id: str) -> dict[str, Any]:
        return _job_view(service, call(service.cancel, job_id), [], tail=0)

    @server.tool(description="List jobs, newest last, optionally filtered by state.")
    def list_jobs(state: str | None = None, limit: Positive = 20) -> dict[str, Any]:
        service.reconcile()
        jobs = service.store.all()
        if state is not None:
            jobs = [j for j in jobs if j.state.value == state]
        return {
            "jobs": [
                {
                    "job_id": j.id,
                    "name": j.spec.name,
                    "backend": j.spec.backend,
                    "state": j.state.value,
                    "gpus": j.gpus,
                    "duration_seconds": j.duration_seconds,
                    "cost_usd": j.cost_usd,
                }
                for j in jobs[-limit:]
            ]
        }
