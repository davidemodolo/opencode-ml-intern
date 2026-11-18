import json
import sys

import pytest
from mcp import Client

from ml_intern_mcp.compute.backend import ComputeError
from ml_intern_mcp.compute.local import LocalBackend
from ml_intern_mcp.compute.models import Hardware, Job, JobSpec, JobState, Offer
from ml_intern_mcp.compute.service import ComputeService
from ml_intern_mcp.compute.store import JobStore
from ml_intern_mcp.server import build_server


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeRemote:
    """A provider whose jobs move through the states the test sets in self.states."""

    name = "remote"
    queues_locally = False
    poll_seconds = 30.0

    def __init__(self, store: JobStore, clock: Clock):
        self.store = store
        self.clock = clock
        self.states: dict[str, JobState] = {}
        self.polls: list[str] = []
        self.cancelled: list[str] = []
        self.fail_submit: Exception | None = None
        self.fail_status: Exception | None = None

    def hardware(self) -> Hardware:
        return Hardware(backend=self.name, offers=[Offer("a10g", "A10G", 1, 24.0, 1.8)])

    def validate(self, spec: JobSpec) -> None:
        if spec.hardware != "a10g":
            raise ComputeError("unknown hardware")

    def price(self, spec: JobSpec) -> float | None:
        return 1.8

    def submit(self, job: Job, gpus: list[int]) -> None:
        if self.fail_submit:
            raise self.fail_submit
        job.remote_id = f"r-{job.id}"
        self.states[job.remote_id] = JobState.QUEUED

    def status(self, job: Job) -> None:
        self.polls.append(job.id)
        if self.fail_status:
            raise self.fail_status
        job.state = self.states[job.remote_id]
        if job.state != JobState.QUEUED and job.started_at is None:
            job.started_at = self.clock()
        if job.state.finished:
            job.finished_at = self.clock()

    def logs(self, job: Job) -> None:
        self.store.write_log(job.id, f"log of {job.remote_id}\n")

    def cancel(self, job: Job) -> None:
        self.cancelled.append(job.remote_id)
        job.state = JobState.CANCELLED
        job.finished_at = self.clock()


def make(tmp_path, clock=None):
    clock = clock or Clock()
    store = JobStore(tmp_path)
    remote = FakeRemote(store, clock)
    local = LocalBackend(store, [sys.executable], gpu_probe=list)
    service = ComputeService(store, {"local": local, "remote": remote}, clock=clock)
    return service, remote, clock


def remote_spec(**kwargs) -> JobSpec:
    return JobSpec(name="r", script="print(1)", backend="remote", hardware="a10g", **kwargs)


def event(tmp_path, job_id):
    path = tmp_path / "events" / f"{job_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def test_remote_job_goes_to_the_provider_at_submit(tmp_path):
    service, _, _ = make(tmp_path)
    job = service.submit(remote_spec())
    assert job.remote_id == f"r-{job.id}" and job.state == JobState.QUEUED
    assert job.usd_per_hour == 1.8
    assert service.get(job.id).remote_id == job.remote_id


def test_remote_jobs_leave_the_local_queue_free(tmp_path):
    service, remote, _ = make(tmp_path)
    remote_job = service.submit(remote_spec())
    remote.states[remote_job.remote_id] = JobState.RUNNING
    service.reconcile(force={remote_job.id})
    local_job = service.submit(JobSpec(name="cpu", script="import time; time.sleep(2)"))
    assert service.get(local_job.id).state == JobState.RUNNING


def test_remote_jobs_are_polled_every_poll_seconds(tmp_path):
    service, remote, clock = make(tmp_path)
    job = service.submit(remote_spec())
    service.reconcile()
    service.reconcile()
    assert remote.polls == [job.id]
    clock.now += 29
    service.reconcile()
    assert remote.polls == [job.id]
    clock.now += 1
    service.reconcile()
    service.reconcile(force={job.id})
    assert remote.polls == [job.id] * 3


def test_finished_remote_job_saves_logs_cost_and_event(tmp_path):
    service, remote, clock = make(tmp_path)
    job = service.submit(remote_spec())
    remote.states[job.remote_id] = JobState.RUNNING
    service.reconcile(force={job.id})
    assert service.get(job.id).state == JobState.RUNNING and event(tmp_path, job.id) is None
    clock.now += 1800
    remote.states[job.remote_id] = JobState.SUCCEEDED
    service.reconcile(force={job.id})
    job = service.get(job.id)
    assert job.state == JobState.SUCCEEDED and job.cost_usd == 0.9
    saved = event(tmp_path, job.id)
    assert saved["cost_usd"] == 0.9 and saved["backend"] == "remote" and saved["hardware"] == "a10g"
    assert saved["log_tail"] == [f"log of {job.remote_id}"]
    remote.polls.clear()
    service.reconcile(force={job.id})
    assert remote.polls == []


def test_a_finished_job_is_saved_when_its_logs_cannot_be_fetched(tmp_path):
    service, remote, _ = make(tmp_path)
    job = service.submit(remote_spec())
    remote.states[job.remote_id] = JobState.SUCCEEDED

    def fail(job):
        raise ConnectionError("logs expired")

    remote.logs = fail
    service.reconcile(force={job.id})
    assert service.get(job.id).state == JobState.SUCCEEDED
    assert event(tmp_path, job.id)["state"] == "succeeded"


def test_running_remote_logs_are_fetched_on_read(tmp_path):
    service, _, _ = make(tmp_path)
    job = service.submit(remote_spec())
    assert service.log_lines(job.id) == [f"log of {job.remote_id}"]


def test_a_failed_submission_fails_the_job_without_an_event(tmp_path):
    service, remote, _ = make(tmp_path)
    remote.fail_submit = RuntimeError("quota exceeded")
    with pytest.raises(ComputeError, match="could not submit the job to remote: quota exceeded"):
        service.submit(remote_spec())
    [job] = service.store.all()
    assert job.state == JobState.FAILED and event(tmp_path, job.id) is None


def test_validation_errors_create_no_job(tmp_path):
    service, _, _ = make(tmp_path)
    with pytest.raises(ComputeError, match="unknown hardware"):
        service.submit(JobSpec(name="r", script="x", backend="remote"))
    assert service.store.all() == []


def test_provider_errors_while_polling_keep_the_job(tmp_path):
    service, remote, _ = make(tmp_path)
    job = service.submit(remote_spec())
    remote.fail_status = ConnectionError("network down")
    service.reconcile(force={job.id})
    assert service.get(job.id).state == JobState.QUEUED


def test_a_restarted_server_picks_up_remote_jobs(tmp_path):
    clock = Clock()
    service, _, _ = make(tmp_path, clock)
    job = service.submit(remote_spec())
    restarted, remote_after, _ = make(tmp_path, clock)
    remote_after.states[job.remote_id] = JobState.FAILED
    restarted.reconcile()
    assert restarted.get(job.id).state == JobState.FAILED
    assert event(tmp_path, job.id)["state"] == "failed"


def test_cancel_stops_the_remote_job(tmp_path):
    service, remote, _ = make(tmp_path)
    job = service.submit(remote_spec())
    cancelled = service.cancel(job.id)
    assert cancelled.state == JobState.CANCELLED and remote.cancelled == [job.remote_id]
    assert event(tmp_path, job.id)["state"] == "cancelled"


def test_cancel_during_submission_cancels_on_the_provider(tmp_path):
    service, remote, _ = make(tmp_path)
    original = remote.submit

    def submit_and_cancel(job, gpus):
        original(job, gpus)
        service.cancel(job.id)

    remote.submit = submit_and_cancel
    job = service.submit(remote_spec())
    assert job.state == JobState.CANCELLED
    assert remote.cancelled[-1] == f"r-{job.id}"


def test_jobs_of_a_backend_no_longer_configured_are_skipped(tmp_path):
    service, _, _ = make(tmp_path)
    job = service.submit(remote_spec())
    store = service.store
    local_only = ComputeService(store, {"local": LocalBackend(store, [sys.executable], list)})
    local_only.reconcile()
    assert local_only.get(job.id).state == JobState.QUEUED


def test_hardware_writes_the_price_catalog(tmp_path):
    service, _, _ = make(tmp_path)
    service.hardware()
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert catalog["backends"]["remote"] == {
        "gpus": 0,
        "status": "ready",
        "usd_per_hour": {"a10g": 1.8},
    }
    assert catalog["backends"]["local"]["usd_per_hour"] == {}


async def test_remote_jobs_over_mcp(tmp_path):
    service, remote, clock = make(tmp_path)
    async with Client(build_server([], service)) as client:
        listed = (await client.call_tool("list_hardware", {})).structured_content
        submitted = await client.call_tool(
            "submit_job",
            {"name": "r", "script": "print(1)", "backend": "remote", "hardware": "a10g"},
        )
        job_id = submitted.structured_content["job_id"]
        queued = (await client.call_tool("job_status", {"job_id": job_id})).structured_content
        remote.states[f"r-{job_id}"] = JobState.SUCCEEDED
        clock.now += 60
        done = (await client.call_tool("job_status", {"job_id": job_id})).structured_content
        jobs = (await client.call_tool("list_jobs", {})).structured_content["jobs"]
    remote_entry = next(b for b in listed["backends"] if b["backend"] == "remote")
    assert "more_offers" not in remote_entry
    assert remote_entry["offers"][0] == {
        "hardware": "a10g",
        "gpu": "A10G",
        "gpu_count": 1,
        "vram_gb": 24.0,
        "usd_per_hour": 1.8,
    }
    assert submitted.structured_content["usd_per_hour"] == 1.8
    assert queued["state"] == "queued" and queued["note"] == "Waiting for remote to start the job."
    assert done["state"] == "succeeded" and done["log_tail"] == [f"log of r-{job_id}"]
    assert jobs[0]["backend"] == "remote" and jobs[0]["cost_usd"] == done["cost_usd"]


async def test_list_hardware_filters_and_caps_the_offers(tmp_path):
    service, remote, _ = make(tmp_path)
    offers = [Offer(f"T4:{n}", "T4", n, 16.0, n * 0.5) for n in range(1, 46)]
    remote.hardware = lambda: Hardware(
        backend="remote", offers=[*offers, Offer("a10g", "A10G", 1, 24.0, 1.8)]
    )
    async with Client(build_server([], service)) as client:
        everything = (await client.call_tool("list_hardware", {})).structured_content
        a10g = (await client.call_tool("list_hardware", {"gpu": "a10"})).structured_content
    remote_all = everything["backends"][1]
    assert len(remote_all["offers"]) == 40 and remote_all["more_offers"] == 6
    assert [o["hardware"] for o in a10g["backends"][1]["offers"]] == ["a10g"]
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert len(catalog["backends"]["remote"]["usd_per_hour"]) == 46
