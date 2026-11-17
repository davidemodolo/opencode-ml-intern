import fcntl
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from ml_intern_mcp.compute.local import LocalBackend
from ml_intern_mcp.compute.models import Hardware, Job, JobSpec, JobState
from ml_intern_mcp.compute.store import JobStore

MAX_TIMEOUT_MINUTES = 24 * 60
CPU_SLOT = -1


class ComputeError(ValueError):
    pass


class ComputeService:
    """Queues jobs per GPU, starts them on a backend and records completion events."""

    def __init__(self, store: JobStore, backends: dict[str, LocalBackend]):
        self.store = store
        self.backends = backends
        self._thread_lock = threading.Lock()
        self._lock_path = store.root / "scheduler.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Serializes scheduling across threads and across servers sharing the data folder."""
        with self._thread_lock, open(self._lock_path, "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            yield

    def hardware(self) -> list[Hardware]:
        return [backend.hardware() for backend in self.backends.values()]

    def submit(self, spec: JobSpec) -> Job:
        backend = self._backend(spec.backend)
        if not spec.script.strip():
            raise ComputeError("script is empty")
        if not 1 <= spec.timeout_minutes <= MAX_TIMEOUT_MINUTES:
            raise ComputeError(f"timeout_minutes must be between 1 and {MAX_TIMEOUT_MINUTES}")
        gpus = {gpu.index: gpu for gpu in backend.hardware().gpus}
        if spec.gpu is not None and spec.gpu not in gpus:
            raise ComputeError(f"gpu {spec.gpu} does not exist on backend {spec.backend}")
        if spec.vram_gb is not None:
            candidates = [gpus[spec.gpu]] if spec.gpu is not None else list(gpus.values())
            if not candidates:
                raise ComputeError("vram_gb was given but this backend has no GPU")
            largest = max(gpu.memory_total_gb for gpu in candidates)
            if spec.vram_gb > largest:
                raise ComputeError(
                    f"the job needs {spec.vram_gb} GB of VRAM but the largest GPU has {largest} GB"
                )
        with self._locked():
            job = self.store.create(spec)
            self._schedule()
            return self.store.get(job.id) or job

    def get(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise ComputeError(f"job {job_id} does not exist")
        return job

    def cancel(self, job_id: str) -> Job:
        with self._locked():
            job = self.get(job_id)
            if job.state.finished:
                return job
            self._backend(job.spec.backend).cancel(job)
            self.store.save(job)
            self.store.write_event(job)
            self._schedule()
            return job

    def reconcile(self) -> None:
        with self._locked():
            for job in self.store.all():
                if job.state != JobState.RUNNING:
                    continue
                self._backend(job.spec.backend).refresh(job)
                if job.state.finished:
                    self.store.save(job)
                    self.store.write_event(job)
            self._schedule()

    def _schedule(self) -> None:
        jobs = self.store.all()
        busy = {self._slot(j.gpu) for j in jobs if j.state == JobState.RUNNING}
        for job in jobs:
            if job.state != JobState.QUEUED:
                continue
            backend = self._backend(job.spec.backend)
            gpu = self._pick_gpu(backend, job.spec, busy)
            if gpu is False:
                continue
            try:
                backend.start(job, gpu)
            except OSError as error:
                job.state = JobState.FAILED
                job.error = f"could not start the job: {error}"
                self.store.save(job)
                self.store.write_event(job)
                continue
            busy.add(self._slot(gpu))
            self.store.save(job)

    def _pick_gpu(self, backend: LocalBackend, spec: JobSpec, busy: set[int]) -> int | None | bool:
        gpus = backend.hardware().gpus
        if not gpus:
            return None if CPU_SLOT not in busy else False
        candidates = [g for g in gpus if spec.gpu is None or g.index == spec.gpu]
        free = [g for g in candidates if g.index not in busy]
        if spec.vram_gb is not None:
            free = [g for g in free if g.memory_free_gb >= spec.vram_gb]
        if not free:
            return False
        return max(free, key=lambda g: g.memory_free_gb).index

    @staticmethod
    def _slot(gpu: int | None) -> int:
        return CPU_SLOT if gpu is None else gpu

    def _backend(self, name: str) -> LocalBackend:
        backend = self.backends.get(name)
        if backend is None:
            raise ComputeError(f"unknown backend {name!r}; available: {sorted(self.backends)}")
        return backend
