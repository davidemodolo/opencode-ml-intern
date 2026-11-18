import json
from datetime import UTC, datetime

import pytest
from huggingface_hub._jobs_api import JobHardwareInfo, JobInfo

from ml_intern_mcp.compute.backend import ComputeError
from ml_intern_mcp.compute.hf_jobs import HfJobsBackend
from ml_intern_mcp.compute.models import JobSpec, JobState
from ml_intern_mcp.compute.service import ComputeService
from ml_intern_mcp.compute.store import JobStore
from ml_intern_mcp.config import HfJobsConfig

START = datetime(2026, 11, 18, 10, 0, tzinfo=UTC)
API_TIME = "%Y-%m-%dT%H:%M:%SZ"


def hardware_info(name, cost_per_minute, accelerator=None):
    return JobHardwareInfo(
        name=name,
        prettyName=name,
        cpu="8 vCPU",
        ram="32 GB",
        accelerator=accelerator,
        unitCostMicroUSD=int(cost_per_minute * 1e6),
        unitCostUSD=cost_per_minute,
        unitLabel="minute",
    )


A10G = {"type": "gpu", "model": "A10G", "quantity": "1", "vram": "24 GB", "manufacturer": "Nvidia"}


def job_info(stage, message=None, started=None, finished=None):
    return JobInfo(
        id="hf-1",
        owner={"id": "u1", "name": "alice", "type": "user"},
        status={"stage": stage, "message": message},
        startedAt=started.strftime(API_TIME) if started else None,
        finishedAt=finished.strftime(API_TIME) if finished else None,
    )


class FakeHfApi:
    def __init__(self):
        self.info = job_info("SCHEDULING")
        self.runs = []
        self.cancelled = []

    def list_jobs_hardware(self):
        return [hardware_info("cpu-basic", 0.0002), hardware_info("a10g-small", 0.0167, A10G)]

    def run_uv_job(self, **kwargs):
        self.runs.append(kwargs)
        return self.info

    def inspect_job(self, *, job_id, namespace=None):
        assert job_id == "hf-1"
        return self.info

    def fetch_job_logs(self, *, job_id, namespace=None):
        return iter(["step 1 loss 0.5\n", "done"])

    def cancel_job(self, *, job_id, namespace=None):
        self.cancelled.append(job_id)


@pytest.fixture
def setup(tmp_path):
    store = JobStore(tmp_path)
    api = FakeHfApi()
    backend = HfJobsBackend(store, HfJobsConfig(namespace="org"), api=api, token=lambda: "hf_x")
    service = ComputeService(store, {"hf_jobs": backend})
    return service, backend, api


def spec(**kwargs):
    values = {"name": "sft", "script": "print(1)", "backend": "hf_jobs", "hardware": "a10g-small"}
    return JobSpec(**{**values, **kwargs})


def test_catalog_prices_are_per_hour(setup):
    _, backend, _ = setup
    offers = {o.hardware: o for o in backend.hardware().offers}
    assert offers["a10g-small"].usd_per_hour == pytest.approx(1.002)
    assert offers["a10g-small"].vram_gb == 24.0 and offers["a10g-small"].gpu == "A10G"
    assert offers["cpu-basic"].gpu is None and offers["cpu-basic"].gpu_count == 0


def test_status_reports_a_missing_token(tmp_path):
    backend = HfJobsBackend(JobStore(tmp_path), HfJobsConfig(), api=FakeHfApi(), token=lambda: None)
    assert "hf auth login" in backend.hardware().status
    with pytest.raises(ComputeError, match="HF token"):
        backend.validate(spec())


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"hardware": None}, "one of: a10g-small, cpu-basic"),
        ({"hardware": "h100"}, "one of"),
        ({"gpus": 2}, "local backend"),
        ({"vram_gb": 40}, "a10g-small has 24.0 GB"),
    ],
)
def test_invalid_specs_fail_fast(setup, bad, message):
    service, _, api = setup
    with pytest.raises(ComputeError, match=message):
        service.submit(spec(**bad))
    assert api.runs == []


def test_submit_runs_the_script_with_uv(setup):
    service, _, api = setup
    job = service.submit(spec(timeout_minutes=30, env={"MAX_STEPS": "10"}))
    [run] = api.runs
    assert run["script"] == str(service.store.folder(job.id) / "script.py")
    assert run["flavor"] == "a10g-small" and run["timeout"] == 1800 and run["namespace"] == "org"
    assert run["env"] == {"MAX_STEPS": "10", "PYTHONUNBUFFERED": "1"}
    assert run["secrets"] == {"HF_TOKEN": "hf_x"} and run["labels"] == {"ml-intern-job": job.id}
    assert job.remote_id == "hf-1" and job.state == JobState.QUEUED
    assert job.usd_per_hour == pytest.approx(1.002)


def test_a_completed_job_records_duration_cost_logs_and_event(setup):
    service, _, api = setup
    job = service.submit(spec())
    api.info = job_info("RUNNING", started=START)
    service.reconcile(force={job.id})
    assert service.get(job.id).state == JobState.RUNNING
    api.info = job_info("COMPLETED", started=START, finished=START.replace(hour=11))
    service.reconcile(force={job.id})
    job = service.get(job.id)
    assert job.state == JobState.SUCCEEDED and job.duration_seconds == 3600
    assert job.cost_usd == pytest.approx(1.002)
    assert service.store.log_lines(job.id) == ["step 1 loss 0.5", "done"]
    event = json.loads((service.store.events_dir / f"{job.id}.json").read_text())
    assert event["state"] == "succeeded" and event["log_tail"] == ["step 1 loss 0.5", "done"]


@pytest.mark.parametrize(
    ("minutes", "state"),
    [(30, JobState.FAILED), (60, JobState.TIMED_OUT), (90, JobState.TIMED_OUT)],
)
def test_errors_after_the_timeout_are_timeouts(setup, minutes, state):
    service, _, api = setup
    job = service.submit(spec(timeout_minutes=60))
    finished = START.replace(hour=10 + minutes // 60, minute=minutes % 60)
    api.info = job_info("ERROR", message="exit 1", started=START, finished=finished)
    service.reconcile(force={job.id})
    job = service.get(job.id)
    assert job.state == state and job.error == "exit 1"


def test_cancel_and_provider_cancellation(setup):
    service, _, api = setup
    job = service.submit(spec())
    service.cancel(job.id)
    assert api.cancelled == ["hf-1"]
    other = service.submit(spec())
    api.info = job_info("CANCELED")
    service.reconcile(force={other.id})
    assert service.get(other.id).state == JobState.CANCELLED
