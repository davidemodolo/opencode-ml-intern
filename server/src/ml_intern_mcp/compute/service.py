import fcntl
import logging
import threading
import time
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager

from ml_intern_mcp.compute.backend import Backend, ComputeError
from ml_intern_mcp.compute.models import Gpu, Hardware, Job, JobSpec, JobState
from ml_intern_mcp.compute.store import JobStore

MAX_TIMEOUT_MINUTES = 24 * 60
CPU_SLOT = -1

logger = logging.getLogger(__name__)

__all__ = ["ComputeError", "ComputeService"]


class ComputeService:
    """Queues local jobs per GPU, hands remote jobs to their provider and records completion events."""

    def __init__(
        self,
        store: JobStore,
        backends: dict[str, Backend],
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.backends = backends
        self.clock = clock
        self._thread_lock = threading.Lock()
        self._lock_path = store.root / "scheduler.lock"
        self._polled: dict[str, float] = {}

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Serializes scheduling across threads and across servers sharing the data folder."""
        with self._thread_lock, open(self._lock_path, "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            yield

    def hardware(self) -> list[Hardware]:
        hardware = [backend.hardware() for backend in self.backends.values()]
        with self._locked():
            self.store.write_catalog(hardware)
        return hardware

    def submit(self, spec: JobSpec) -> Job:
        backend = self._backend(spec.backend)
        if not spec.script.strip():
            raise ComputeError("script is empty")
        if not 1 <= spec.timeout_minutes <= MAX_TIMEOUT_MINUTES:
            raise ComputeError(f"timeout_minutes must be between 1 and {MAX_TIMEOUT_MINUTES}")
        if spec.gpus < 1:
            raise ComputeError("gpus must be at least 1")
        backend.validate(spec)
        with self._locked():
            job = self.store.create(spec)
            job.usd_per_hour = backend.price(spec)
            self.store.save(job)
            if backend.queues_locally:
                self._schedule()
                return self.store.get(job.id) or job
        return self._submit_remote(backend, job)

    def _submit_remote(self, backend: Backend, job: Job) -> Job:
        try:
            backend.submit(job, [])
        except Exception as error:
            with self._locked():
                job.state = JobState.FAILED
                job.error = f"could not submit the job to {backend.name}: {error}"
                job.finished_at = self.clock()
                self.store.save(job)
            raise ComputeError(job.error) from error
        with self._locked():
            current = self.store.get(job.id)
            if current is not None and current.state == JobState.CANCELLED:
                backend.cancel(job)
                return current
            self.store.save(job)
            return job

    def get(self, job_id: str) -> Job:
        job = self.store.get(job_id)
        if job is None:
            raise ComputeError(f"job {job_id} does not exist")
        return job

    def log_lines(self, job_id: str) -> list[str]:
        """Returns the job log, first fetching it from the provider for a running remote job."""
        job = self.get(job_id)
        backend = self.backends.get(job.spec.backend)
        if backend is not None and self._remote_active(backend, job):
            try:
                backend.logs(job)
            except Exception:
                logger.warning("could not fetch the logs of job %s", job.id, exc_info=True)
        return self.store.log_lines(job_id)

    def cancel(self, job_id: str) -> Job:
        with self._locked():
            job = self.get(job_id)
            if job.state.finished:
                return job
            self._backend(job.spec.backend).cancel(job)
            self._finish(job)
            self._schedule()
            return job

    def reconcile(self, force: Collection[str] = ()) -> None:
        """Updates running jobs, starts queued local jobs and polls remote jobs that are due.

        Remote jobs are polled every poll_seconds of their backend; job IDs in force are polled now.
        """
        with self._locked():
            jobs = self.store.all()
            for job in jobs:
                backend = self.backends.get(job.spec.backend)
                if backend is None or not backend.queues_locally or job.state != JobState.RUNNING:
                    continue
                backend.status(job)
                if job.state.finished:
                    self._finish(job)
            self._schedule(jobs)
        for job in jobs:
            self._poll(job, force=job.id in force)

    def _poll(self, job: Job, force: bool) -> None:
        backend = self.backends.get(job.spec.backend)
        if backend is None or not self._remote_active(backend, job):
            return
        now = self.clock()
        if not force and now - self._polled.get(job.id, 0.0) < backend.poll_seconds:
            return
        self._polled[job.id] = now
        try:
            backend.status(job)
        except Exception:
            logger.warning("could not poll job %s on %s", job.id, backend.name, exc_info=True)
            return
        if job.state.finished:
            try:
                backend.logs(job)
            except Exception:
                logger.warning("could not fetch the logs of job %s", job.id, exc_info=True)
        with self._locked():
            current = self.store.get(job.id)
            if current is None or current.state.finished:
                return
            if job.state.finished:
                self._polled.pop(job.id, None)
                self._finish(job)
            else:
                self.store.save(job)

    @staticmethod
    def _remote_active(backend: Backend, job: Job) -> bool:
        return not backend.queues_locally and not job.state.finished and job.remote_id is not None

    def _finish(self, job: Job) -> None:
        if job.cost_usd is None and job.usd_per_hour is not None:
            job.cost_usd = round(job.usd_per_hour * (job.duration_seconds or 0) / 3600, 4)
        self.store.save(job)
        self.store.write_event(job)

    def _schedule(self, jobs: list[Job] | None = None) -> None:
        jobs = self.store.all() if jobs is None else jobs
        busy = {
            slot
            for j in jobs
            if j.state == JobState.RUNNING and self._queues_locally(j)
            for slot in self._slots(j)
        }
        gpus_by_backend: dict[str, list[Gpu]] = {}
        for job in jobs:
            backend = self.backends.get(job.spec.backend)
            if backend is None or not backend.queues_locally or job.state != JobState.QUEUED:
                continue
            if backend.name not in gpus_by_backend:
                gpus_by_backend[backend.name] = backend.hardware().gpus
            gpus = self._pick_gpus(gpus_by_backend[backend.name], job.spec, busy)
            if gpus is None:
                continue
            try:
                backend.submit(job, gpus)
            except OSError as error:
                job.state = JobState.FAILED
                job.error = f"could not start the job: {error}"
                self._finish(job)
                continue
            busy.update(self._slots(job))
            self.store.save(job)

    @staticmethod
    def _pick_gpus(gpus: list[Gpu], spec: JobSpec, busy: set[int]) -> list[int] | None:
        """Returns the GPUs to start the job on, [] for a CPU job, or None when it must wait."""
        if not gpus:
            return [] if CPU_SLOT not in busy else None
        free = [
            g for g in gpus if g.index not in busy and (spec.gpu is None or g.index == spec.gpu)
        ]
        chosen = sorted(free, key=lambda g: g.memory_free_gb, reverse=True)[: spec.gpus]
        if len(chosen) < spec.gpus:
            return None
        if spec.vram_gb is not None and sum(g.memory_free_gb for g in chosen) < spec.vram_gb:
            return None
        return sorted(g.index for g in chosen)

    def _queues_locally(self, job: Job) -> bool:
        backend = self.backends.get(job.spec.backend)
        return backend is not None and backend.queues_locally

    @staticmethod
    def _slots(job: Job) -> list[int]:
        return job.gpus or [CPU_SLOT]

    def _backend(self, name: str) -> Backend:
        backend = self.backends.get(name)
        if backend is None:
            raise ComputeError(f"unknown backend {name!r}; available: {sorted(self.backends)}")
        return backend
