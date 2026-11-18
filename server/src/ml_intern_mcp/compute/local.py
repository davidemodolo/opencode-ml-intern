import os
import shlex
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from ml_intern_mcp.compute.backend import ComputeError
from ml_intern_mcp.compute.models import Gpu, Hardware, Job, JobSpec, JobState
from ml_intern_mcp.compute.store import JobStore

TIMEOUT_EXIT_CODE = 124
SIGKILL_EXIT_CODE = 137
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
        return [
            int(child) for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        ]
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
    queues_locally = True
    poll_seconds = 0.0

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

    def validate(self, spec: JobSpec) -> None:
        if spec.hardware is not None:
            raise ComputeError("hardware applies to remote backends; the local backend takes gpus")
        gpus = {gpu.index: gpu for gpu in self.gpu_probe()}
        if spec.gpu is not None:
            if spec.gpus != 1:
                raise ComputeError("set either gpu, to pick one GPU, or gpus, to use several")
            if spec.gpu not in gpus:
                raise ComputeError(f"gpu {spec.gpu} does not exist on backend {self.name}")
        if spec.gpus > max(len(gpus), 1):
            raise ComputeError(
                f"the job asks for {spec.gpus} GPUs but backend {self.name} has {len(gpus)}"
            )
        if spec.vram_gb is None:
            return
        candidates = [gpus[spec.gpu]] if spec.gpu is not None else list(gpus.values())
        if not candidates:
            raise ComputeError("vram_gb was given but this backend has no GPU")
        totals = sorted((gpu.memory_total_gb for gpu in candidates), reverse=True)
        largest = round(sum(totals[: spec.gpus]), 2)
        if spec.vram_gb > largest:
            where = (
                "the largest GPU has" if spec.gpus == 1 else f"the {spec.gpus} largest GPUs have"
            )
            raise ComputeError(f"the job needs {spec.vram_gb} GB of VRAM but {where} {largest} GB")

    def price(self, spec: JobSpec) -> float | None:
        return None

    def submit(self, job: Job, gpus: list[int]) -> None:
        folder = self.store.folder(job.id)
        timeout_seconds = job.spec.timeout_minutes * 60
        command = " ".join(shlex.quote(part) for part in [*self.runner, "script.py"])
        wrapper = (
            f"timeout --signal=TERM --kill-after={KILL_GRACE_SECONDS} {timeout_seconds} "
            f"{command} > stdout.log 2>&1; echo $? > exit_code.tmp && mv exit_code.tmp exit_code"
        )
        env = {**os.environ, **job.spec.env, "PYTHONUNBUFFERED": "1"}
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
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
        job.gpus = list(gpus)
        job.state = JobState.RUNNING
        job.started_at = time.time()

    def status(self, job: Job) -> None:
        process = self._processes.get(job.id)
        if process is not None:
            process.poll()
        exit_code = self.store.read_exit_code(job.id)
        if exit_code is not None:
            self._finish(job, exit_code, self.store.exit_code_path(job.id).stat().st_mtime)
            return
        if job.pid is not None and not _alive(job.pid):
            job.state = JobState.FAILED
            job.error = "the job process ended without writing an exit code"
            job.finished_at = time.time()
            self._processes.pop(job.id, None)

    def logs(self, job: Job) -> None:
        return None

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

    def _finish(self, job: Job, exit_code: int, finished_at: float) -> None:
        job.exit_code = exit_code
        job.finished_at = finished_at
        if exit_code == 0:
            job.state = JobState.SUCCEEDED
        elif exit_code == TIMEOUT_EXIT_CODE or (
            exit_code == SIGKILL_EXIT_CODE
            and (job.duration_seconds or 0) >= job.spec.timeout_minutes * 60
        ):
            job.state = JobState.TIMED_OUT
        else:
            job.state = JobState.FAILED
            if exit_code == SIGKILL_EXIT_CODE:
                job.error = (
                    "the script was killed by SIGKILL, most likely for running out of memory"
                )
        self._processes.pop(job.id, None)
