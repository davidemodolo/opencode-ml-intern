import base64
import json
import math
import sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_intern_mcp.compute.backend import ComputeError
from ml_intern_mcp.compute.models import JobSpec, JobState
from ml_intern_mcp.compute.service import ComputeService
from ml_intern_mcp.compute.skypilot import (
    SkyPilotBackend,
    SkyPilotClient,
    bridge_runner,
    exit_code_from_log,
    offers_from_catalog,
    task_config,
)
from ml_intern_mcp.compute.store import JobStore
from ml_intern_mcp.config import SkyPilotConfig

Info = namedtuple(
    "Info",
    "cloud instance_type accelerator_name accelerator_count cpu_count device_memory memory "
    "price spot_price region",
)


def info(cloud, accelerator, count, price, spot=math.nan, memory=80.0):
    return Info(cloud, "x", accelerator, count, 8, memory, 64, price, spot, "r")


CATALOG = {
    "A100": [
        info("RunPod", "A100", 1, 1.6, spot=0.9),
        info("Vast", "A100", 1, 1.2),
        info("RunPod", "A100", 2, 3.2, spot=1.8),
    ],
    "H100": [info("RunPod", "H100", 1, math.nan)],
}


class FakeSky:
    def __init__(self):
        self.launched = []
        self.record = None
        self.error = None
        self.log = ""
        self.cancelled = []

    def enabled_infra(self):
        return ["runpod", "vast"]

    def accelerators(self, infra):
        return CATALOG

    def launch(self, config, name):
        self.launched.append((config, name))
        return "req-1"

    def job(self, name):
        return self.record

    def request_error(self, request_id):
        assert request_id == "req-1"
        return self.error

    def logs(self, name):
        return self.log

    def cancel(self, name):
        self.cancelled.append(name)


@pytest.fixture
def setup(tmp_path):
    store = JobStore(tmp_path)
    sky = FakeSky()
    backend = SkyPilotBackend(store, SkyPilotConfig(), client_factory=lambda: sky)
    return ComputeService(store, {"skypilot": backend}), backend, sky


def spec(**kwargs):
    values = {"name": "sft", "script": "print('hi')", "backend": "skypilot", "hardware": "A100:1"}
    return JobSpec(**{**values, **kwargs})


def record(status, start=1000.0, end=None, reason=None):
    return SimpleNamespace(
        job_name="x", job_id=1, status=status, start_at=start, end_at=end, failure_reason=reason
    )


def test_offers_keep_the_cheapest_priced_instance():
    offers = {o.hardware: o for o in offers_from_catalog(CATALOG, use_spot=False)}
    assert offers.keys() == {"A100:1", "A100:2"}
    assert offers["A100:1"].usd_per_hour == 1.2 and offers["A100:1"].vram_gb == 80.0
    spot = {o.hardware: o.usd_per_hour for o in offers_from_catalog(CATALOG, use_spot=True)}
    assert spot == {"A100:1": 0.9, "A100:2": 1.8}


def test_task_runs_the_script_with_uv_under_the_timeout(setup):
    service, _, _ = setup
    job = service.store.create(spec(timeout_minutes=90, env={"MAX_STEPS": "5"}))
    config = SkyPilotConfig(infra=["runpod", "vast"], use_spot=True, max_restarts_on_errors=1)
    task = task_config(job, "print('hi')", config, hf_token="hf_x")
    assert task["name"] == f"mli-{job.id}"
    assert task["resources"] == {
        "accelerators": "A100:1",
        "use_spot": True,
        "job_recovery": {"strategy": "EAGER_NEXT_REGION", "max_restarts_on_errors": 1},
        "any_of": [{"infra": "runpod"}, {"infra": "vast"}],
    }
    assert base64.b64decode(task["envs"]["ML_INTERN_SCRIPT"]).decode() == "print('hi')"
    assert task["envs"]["MAX_STEPS"] == "5" and task["secrets"] == {"HF_TOKEN": "hf_x"}
    assert "timeout --signal=TERM --kill-after=30 5400 uv run --script script.py" in task["run"]
    single = task_config(job, "x", SkyPilotConfig(infra=["runpod"]), hf_token=None)
    assert single["resources"]["infra"] == "runpod" and "secrets" not in single


def test_exit_code_is_read_from_the_last_marker():
    assert exit_code_from_log("a\n[ml-intern] exit code 1\nb\n[ml-intern] exit code 124\n") == 124
    assert exit_code_from_log("no marker") is None


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"hardware": "T4:1"}, "such as: A100:1, A100:2"),
        ({"gpus": 2}, "such as A100:2"),
    ],
)
def test_invalid_specs_fail_fast(setup, bad, message):
    service, _, sky = setup
    with pytest.raises(ComputeError, match=message):
        service.submit(spec(**bad))
    assert sky.launched == []


def test_a_job_runs_and_succeeds(setup):
    service, _, sky = setup
    job = service.submit(spec())
    assert sky.launched[0][1] == f"mli-{job.id}" and job.remote_id == "req-1"
    assert job.usd_per_hour == 1.2
    service.reconcile(force={job.id})
    assert service.get(job.id).state == JobState.QUEUED
    sky.record = record("RECOVERING")
    service.reconcile(force={job.id})
    assert service.get(job.id).state == JobState.RUNNING
    sky.record = record("SUCCEEDED", end=1000.0 + 1800)
    sky.log = "loss 0.1\n[ml-intern] exit code 0\n"
    service.reconcile(force={job.id})
    job = service.get(job.id)
    assert job.state == JobState.SUCCEEDED and job.exit_code == 0 and job.cost_usd == 0.6
    event = json.loads((service.store.events_dir / f"{job.id}.json").read_text())
    assert event["log_tail"] == ["loss 0.1", "[ml-intern] exit code 0"]


def test_exit_code_124_is_a_timeout(setup):
    service, _, sky = setup
    job = service.submit(spec())
    sky.record = record("FAILED", end=5000.0, reason="exit 124")
    sky.log = "[ml-intern] exit code 124\n"
    service.reconcile(force={job.id})
    assert service.get(job.id).state == JobState.TIMED_OUT


def test_provider_failures_carry_their_reason(setup):
    service, _, sky = setup
    job = service.submit(spec())
    sky.record = record("FAILED_NO_RESOURCE", start=None, end=1.0, reason="no A100 available")
    service.reconcile(force={job.id})
    job = service.get(job.id)
    assert job.state == JobState.FAILED and job.error == "FAILED_NO_RESOURCE: no A100 available"


def test_a_failed_launch_request_fails_the_job(setup):
    service, _, sky = setup
    job = service.submit(spec())
    sky.error = "controller could not start"
    service.reconcile(force={job.id})
    job = service.get(job.id)
    assert job.state == JobState.FAILED and "controller could not start" in job.error


def test_cancel(setup):
    service, _, sky = setup
    job = service.submit(spec())
    service.cancel(job.id)
    assert sky.cancelled == [f"mli-{job.id}"]


def test_hardware_reports_a_missing_skypilot_install(tmp_path):
    def missing():
        raise ComputeError("the skypilot backend needs the skypilot extra")

    backend = SkyPilotBackend(JobStore(tmp_path), SkyPilotConfig(), client_factory=missing)
    assert backend.hardware().status == "unavailable: the skypilot backend needs the skypilot extra"


def test_no_enabled_infra_is_reported(tmp_path):
    sky = FakeSky()
    sky.enabled_infra = list
    backend = SkyPilotBackend(JobStore(tmp_path), SkyPilotConfig(), client_factory=lambda: sky)
    assert "sky check" in backend.hardware().status


def test_the_bridge_calls_the_sdk_and_returns_json(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent / "fake_sky"))
    monkeypatch.setenv("FAKE_SKY_LOG", str(tmp_path / "calls.jsonl"))
    client = SkyPilotClient(["runpod/eu"], runner=[sys.executable])
    assert client.enabled_infra() == ["runpod", "vast"]
    [a100] = client.accelerators(["runpod/eu", "vast"])["A100"]
    assert a100.price == 1.6 and math.isnan(a100.spot_price)
    assert client.launch({"name": "mli-a"}, "mli-a") == "req-7"
    record = client.job("mli-a")
    assert record.job_id == 2 and record.status == "RUNNING" and record.start_at == 5.0
    assert client.job("mli-z") is None
    assert client.request_error("req-7") == "no quota"
    assert client.logs("mli-a") == "log of mli-a\n"
    client.cancel("mli-a")
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert calls == [
        ["list_accelerators", ["runpod", "vast"]],
        ["launch", {"name": "mli-a"}, "mli-a"],
        ["cancel", "mli-a"],
    ]


def test_bridge_failures_become_compute_errors(tmp_path):
    client = SkyPilotClient([], runner=[sys.executable, "-c", "raise SystemExit('boom')"])
    with pytest.raises(ComputeError, match="SkyPilot enabled_infra failed: boom"):
        client.enabled_infra()


def test_the_bridge_runs_with_the_skypilot_extras_of_the_configured_infra():
    assert bridge_runner(["runpod", "vast/us", "runpod/eu"])[:5] == [
        "uv",
        "run",
        "--no-project",
        "--with",
        "skypilot[runpod,vast]==0.14.3",
    ]
    assert "skypilot==0.14.3" in bridge_runner([])
