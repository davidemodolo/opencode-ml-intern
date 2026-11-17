from dataclasses import asdict
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ml_intern_mcp.compute.models import Job, JobSpec, JobState
from ml_intern_mcp.compute.service import ComputeError, ComputeService

STATUS_TAIL_LINES = 20
RUNNING_NOTE = (
    "The job is still running. Its completion is notified to the session automatically: "
    "do not poll, continue with other work or end your turn."
)


def _job_view(service: ComputeService, job: Job, tail: int = STATUS_TAIL_LINES) -> dict[str, Any]:
    view: dict[str, Any] = {
        "job_id": job.id,
        "name": job.spec.name,
        "backend": job.spec.backend,
        "state": job.state.value,
        "gpu": job.gpu,
        "exit_code": job.exit_code,
        "duration_seconds": job.duration_seconds,
        "timeout_minutes": job.spec.timeout_minutes,
        "error": job.error,
        "log_tail": service.store.log_lines(job.id)[-tail:] if tail else [],
    }
    if job.state == JobState.RUNNING:
        view["note"] = RUNNING_NOTE
    if job.state == JobState.QUEUED:
        view["note"] = "Waiting for a free GPU" + (
            f" with {job.spec.vram_gb} GB of free VRAM." if job.spec.vram_gb else "."
        )
    return view


def register_compute_tools(server: MCPServer, service: ComputeService) -> None:
    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ComputeError as error:
            raise ToolError(str(error)) from error

    @server.tool(
        description=(
            "List the compute backends with their GPUs (name, total and free VRAM in GB) "
            "and hourly cost in USD."
        )
    )
    def list_hardware() -> dict[str, Any]:
        return {"backends": [asdict(h) for h in service.hardware()]}

    @server.tool(
        description=(
            "Submit a Python training or evaluation script as a background job. The script runs "
            "with 'uv run --script', so declare its dependencies inline with a PEP 723 block "
            "('# /// script' ... 'dependencies = [...]' ... '# ///'). Jobs run one per GPU; "
            "extra jobs are queued. Set vram_gb to the VRAM the job needs to fail fast when no "
            "GPU can fit it. Completion is notified to the session automatically."
        )
    )
    def submit_job(
        name: str,
        script: str,
        gpu: int | None = None,
        vram_gb: float | None = None,
        timeout_minutes: int = 60,
        env: dict[str, str] | None = None,
        backend: str = "local",
    ) -> dict[str, Any]:
        spec = JobSpec(
            name=name,
            script=script,
            backend=backend,
            gpu=gpu,
            vram_gb=vram_gb,
            timeout_minutes=timeout_minutes,
            env=env or {},
        )
        job = call(service.submit, spec)
        return _job_view(service, job, tail=0)

    @server.tool(description="Show the state, exit code, duration and last log lines of a job.")
    def job_status(job_id: str) -> dict[str, Any]:
        service.reconcile()
        return _job_view(service, call(service.get, job_id))

    @server.tool(
        description=(
            "Read the log of a job. By default returns the last 'tail' lines; "
            "set offset to read a window of 'limit' lines from that line."
        )
    )
    def job_logs(
        job_id: str, tail: int = 100, offset: int | None = None, limit: int = 200
    ) -> dict[str, Any]:
        call(service.get, job_id)
        lines = service.store.log_lines(job_id)
        selected = lines[offset : offset + limit] if offset is not None else lines[-tail:]
        return {"job_id": job_id, "total_lines": len(lines), "lines": selected}

    @server.tool(description="Cancel a queued or running job.")
    def cancel_job(job_id: str) -> dict[str, Any]:
        return _job_view(service, call(service.cancel, job_id), tail=0)

    @server.tool(description="List jobs, newest last, optionally filtered by state.")
    def list_jobs(state: str | None = None, limit: int = 20) -> dict[str, Any]:
        service.reconcile()
        jobs = service.store.all()
        if state is not None:
            jobs = [j for j in jobs if j.state.value == state]
        return {
            "jobs": [
                {
                    "job_id": j.id,
                    "name": j.spec.name,
                    "state": j.state.value,
                    "gpu": j.gpu,
                    "duration_seconds": j.duration_seconds,
                }
                for j in jobs[-limit:]
            ]
        }
