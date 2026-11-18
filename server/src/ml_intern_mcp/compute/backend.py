from typing import Protocol

from ml_intern_mcp.compute.models import Hardware, Job, JobSpec


class ComputeError(ValueError):
    pass


class Backend(Protocol):
    """A place where jobs run.

    Backends with queues_locally set wait in the server's GPU queue and get the GPUs to use in
    submit; the others are handed to their provider at once and polled every poll_seconds.
    """

    name: str
    queues_locally: bool
    poll_seconds: float

    def hardware(self) -> Hardware: ...

    def validate(self, spec: JobSpec) -> None: ...

    def price(self, spec: JobSpec) -> float | None: ...

    def submit(self, job: Job, gpus: list[int]) -> None: ...

    def status(self, job: Job) -> None: ...

    def logs(self, job: Job) -> None: ...

    def cancel(self, job: Job) -> None: ...
