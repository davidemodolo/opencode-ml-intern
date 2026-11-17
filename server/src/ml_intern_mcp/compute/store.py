import json
import time
import uuid
from pathlib import Path

from ml_intern_mcp.compute.models import Job, JobSpec, JobState
from ml_intern_mcp.files import write_atomic


class JobStore:
    """Persists jobs as folders: job.json, script.py, stdout.log and exit_code."""

    def __init__(self, root: Path):
        self.root = root
        self.jobs_dir = root / "jobs"
        self.events_dir = root / "events"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def create(self, spec: JobSpec) -> Job:
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        job = Job(id=job_id, spec=spec, state=JobState.QUEUED, created_at=time.time())
        folder = self.folder(job_id)
        folder.mkdir(parents=True)
        (folder / "script.py").write_text(spec.script)
        self.save(job)
        return job

    def folder(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def log_path(self, job_id: str) -> Path:
        return self.folder(job_id) / "stdout.log"

    def exit_code_path(self, job_id: str) -> Path:
        return self.folder(job_id) / "exit_code"

    def save(self, job: Job) -> None:
        write_atomic(self.folder(job.id) / "job.json", json.dumps(job.to_dict(), indent=2))

    def get(self, job_id: str) -> Job | None:
        path = self.folder(job_id) / "job.json"
        if not path.exists():
            return None
        return Job.from_dict(json.loads(path.read_text()))

    def all(self) -> list[Job]:
        jobs = [self.get(p.name) for p in self.jobs_dir.iterdir() if p.is_dir()]
        return sorted((j for j in jobs if j is not None), key=lambda j: j.created_at)

    def read_exit_code(self, job_id: str) -> int | None:
        path = self.exit_code_path(job_id)
        if not path.exists():
            return None
        text = path.read_text().strip()
        return int(text) if text else None

    def log_lines(self, job_id: str) -> list[str]:
        path = self.log_path(job_id)
        if not path.exists():
            return []
        return path.read_text(errors="replace").splitlines()

    def write_event(self, job: Job, tail: int = 20) -> None:
        event = {
            "job_id": job.id,
            "name": job.spec.name,
            "backend": job.spec.backend,
            "state": job.state.value,
            "exit_code": job.exit_code,
            "duration_seconds": job.duration_seconds,
            "error": job.error,
            "workdir": str(self.folder(job.id)),
            "log_tail": self.log_lines(job.id)[-tail:],
        }
        write_atomic(self.events_dir / f"{job.id}.json", json.dumps(event, indent=2))
