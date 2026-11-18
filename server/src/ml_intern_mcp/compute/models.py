import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"

    @property
    def finished(self) -> bool:
        return self not in (JobState.QUEUED, JobState.RUNNING)


@dataclass
class Gpu:
    index: int
    name: str
    memory_total_gb: float
    memory_free_gb: float


@dataclass
class Offer:
    """A machine type a remote backend can run a job on, priced per hour."""

    hardware: str
    gpu: str | None
    gpu_count: int
    vram_gb: float | None
    usd_per_hour: float


@dataclass
class Hardware:
    backend: str
    gpus: list[Gpu] = field(default_factory=list)
    offers: list[Offer] = field(default_factory=list)
    status: str = "ready"


@dataclass
class JobSpec:
    name: str
    script: str
    backend: str = "local"
    hardware: str | None = None
    gpu: int | None = None
    gpus: int = 1
    vram_gb: float | None = None
    timeout_minutes: int = 60
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Job:
    id: str
    spec: JobSpec
    state: JobState
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    pid: int | None = None
    gpus: list[int] = field(default_factory=list)
    remote_id: str | None = None
    usd_per_hour: float | None = None
    cost_usd: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        data = dict(data)
        data["spec"] = JobSpec(**data["spec"])
        data["state"] = JobState(data["state"])
        return cls(**data)

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(end - self.started_at, 1)
