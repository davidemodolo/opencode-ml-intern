import os
import shlex
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from ml_intern_mcp.compute.models import Gpu, Hardware, Job, JobState
from ml_intern_mcp.compute.store import JobStore

TIMEOUT_EXIT_CODES = {124, 137}
KILL_GRACE_SECONDS = 30
BYTES_PER_GB = 1024**3


def probe_nvidia_gpus() -> list[Gpu]:
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - any NVML failure means no usable GPU
        return []
    try:
        gpus = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                Gpu(
                    index=index,
                    name=name.decode() if isinstance(name, bytes) else name,
                    memory_total_gb=round(memory.total / BYTES_PER_GB, 2),
                    memory_free_gb=round(memory.free / BYTES_PER_GB, 2),
                )
            )
        return gpus
    finally:
        pynvml.nvmlShutdown()


def _child_pids(pid: int) -> list[int]:
    try:
        return [int(child) for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except OSError:
        return []


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class LocalBackend:
    """Runs jobs as detached process groups on this machine."""

    name = "local"

    def __init__(
        self,
        store: JobStore,
        runner: list[str],
        gpu_probe: Callable[[], list[Gpu]] = probe_nvidia_gpus,
    ):
        self.store = store
        self.runner = runner
        self.gpu_probe = gpu_probe
        self._processes: dict[str, subprocess.Popen] = {}

    def hardware(self) -> Hardware:
        return Hardware(backend=self.name, gpus=self.gpu_probe())

    def start(self, job: Job, gpu: int | None) -> None:
        folder = self.store.folder(job.id)
        timeout_seconds = job.spec.timeout_minutes * 60
        command = " ".join(shlex.quote(part) for part in [*self.runner, "script.py"])
        wrapper = (
            f"timeout --signal=TERM --kill-after={KILL_GRACE_SECONDS} {timeout_seconds} "
            f"{command} > stdout.log 2>&1; echo $? > exit_code.tmp && mv exit_code.tmp exit_code"
        )
        env = {**os.environ, **job.spec.env, "PYTHONUNBUFFERED": "1"}
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(
            ["sh", "-c", wrapper],
            cwd=folder,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._processes[job.id] = process
        job.pid = process.pid
        job.gpu = gpu
        job.state = JobState.RUNNING
        job.started_at = time.time()

    def refresh(self, job: Job) -> None:
        process = self._processes.get(job.id)
        if process is not None:
            process.poll()
        exit_code = self.store.read_exit_code(job.id)
        if exit_code is not None:
            self._finish(job, exit_code)
            return
        if job.pid is not None and not _alive(job.pid):
            job.state = JobState.FAILED
            job.error = "the job process ended without writing an exit code"
            job.finished_at = time.time()
            self._processes.pop(job.id, None)

    def cancel(self, job: Job) -> None:
        if job.pid is not None and _alive(job.pid):
            # timeout(1) moves itself and the script into a process group of its own
            for group in [*_child_pids(job.pid), job.pid]:
                try:
                    os.killpg(group, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        job.state = JobState.CANCELLED
        job.finished_at = time.time()
        self._processes.pop(job.id, None)

    def _finish(self, job: Job, exit_code: int) -> None:
        job.exit_code = exit_code
        job.finished_at = time.time()
        if exit_code == 0:
            job.state = JobState.SUCCEEDED
        elif exit_code in TIMEOUT_EXIT_CODES:
            job.state = JobState.TIMED_OUT
        else:
            job.state = JobState.FAILED
        self._processes.pop(job.id, None)
