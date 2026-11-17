import json
import sys
import time

import pytest
from mcp import Client

from ml_intern_mcp.compute.local import LocalBackend
from ml_intern_mcp.compute.models import Gpu, JobSpec, JobState
from ml_intern_mcp.compute.service import ComputeError, ComputeService
from ml_intern_mcp.compute.store import JobStore
from ml_intern_mcp.server import build_server

ONE_GPU = [Gpu(index=0, name="Test GPU", memory_total_gb=8.0, memory_free_gb=7.5)]


def make_service(root, gpus=ONE_GPU):
    store = JobStore(root)
    backend = LocalBackend(store, [sys.executable], gpu_probe=lambda: list(gpus))
    return ComputeService(store, {"local": backend})


def wait_for(service, job_id, states, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        service.reconcile()
        job = service.get(job_id)
        if job.state in states:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} stayed {service.get(job_id).state}")


FINISHED = {JobState.SUCCEEDED, JobState.FAILED, JobState.TIMED_OUT, JobState.CANCELLED}


def test_job_succeeds_writes_log_and_event(tmp_path):
    service = make_service(tmp_path)
    job = service.submit(JobSpec(name="hello", script="print('hello from job')"))
    job = wait_for(service, job.id, FINISHED)
    assert job.state == JobState.SUCCEEDED and job.exit_code == 0 and job.gpu == 0
    assert service.store.log_lines(job.id) == ["hello from job"]
    event = json.loads((tmp_path / "events" / f"{job.id}.json").read_text())
    assert event["state"] == "succeeded" and event["log_tail"] == ["hello from job"]


def test_job_failure_is_reported(tmp_path):
    service = make_service(tmp_path)
    job = service.submit(JobSpec(name="boom", script="raise SystemExit(3)"))
    job = wait_for(service, job.id, FINISHED)
    assert job.state == JobState.FAILED and job.exit_code == 3


def test_env_and_gpu_are_passed(tmp_path):
    service = make_service(tmp_path)
    script = "import os; print(os.environ['CUDA_VISIBLE_DEVICES'], os.environ['MY_VAR'])"
    job = service.submit(JobSpec(name="env", script=script, env={"MY_VAR": "x"}))
    job = wait_for(service, job.id, FINISHED)
    assert service.store.log_lines(job.id) == ["0 x"]


def test_second_job_waits_for_the_gpu(tmp_path):
    service = make_service(tmp_path)
    first = service.submit(JobSpec(name="slow", script="import time; time.sleep(1.5)"))
    second = service.submit(JobSpec(name="next", script="print('second')"))
    assert service.get(first.id).state == JobState.RUNNING
    assert service.get(second.id).state == JobState.QUEUED
    wait_for(service, first.id, FINISHED)
    second = wait_for(service, second.id, FINISHED)
    assert second.state == JobState.SUCCEEDED


def test_cancel_stops_the_process_group(tmp_path):
    service = make_service(tmp_path)
    job = service.submit(JobSpec(name="forever", script="import time; time.sleep(60)"))
    wait_for(service, job.id, {JobState.RUNNING})
    job = service.cancel(job.id)
    assert job.state == JobState.CANCELLED
    time.sleep(0.5)
    service.reconcile()
    assert service.get(job.id).state == JobState.CANCELLED
    assert (tmp_path / "events" / f"{job.id}.json").exists()


def test_timeout_exit_code_marks_the_job_timed_out(tmp_path):
    store = JobStore(tmp_path)
    backend = LocalBackend(store, [sys.executable], gpu_probe=lambda: list(ONE_GPU))
    job = store.create(JobSpec(name="slow", script="print(1)"))
    job.state = JobState.RUNNING
    store.exit_code_path(job.id).write_text("124")
    backend.refresh(job)
    assert job.state == JobState.TIMED_OUT and job.exit_code == 124


def test_vram_larger_than_any_gpu_fails_fast(tmp_path):
    service = make_service(tmp_path)
    with pytest.raises(ComputeError, match="largest GPU has 8.0 GB"):
        service.submit(JobSpec(name="big", script="print(1)", vram_gb=24))


def test_jobs_survive_a_service_restart(tmp_path):
    service = make_service(tmp_path)
    job = service.submit(JobSpec(name="slow", script="import time; time.sleep(1); print('done')"))
    restarted = make_service(tmp_path)
    job = wait_for(restarted, job.id, FINISHED)
    assert job.state == JobState.SUCCEEDED
    assert restarted.store.log_lines(job.id) == ["done"]


def test_cpu_only_machine_runs_one_job_at_a_time(tmp_path):
    service = make_service(tmp_path, gpus=[])
    first = service.submit(JobSpec(name="a", script="import time; time.sleep(1)"))
    second = service.submit(JobSpec(name="b", script="print('b')"))
    assert service.get(first.id).gpu is None
    assert service.get(second.id).state == JobState.QUEUED
    wait_for(service, second.id, FINISHED)


async def test_compute_tools_over_mcp(tmp_path):
    service = make_service(tmp_path)
    async with Client(build_server([], service)) as client:
        hardware = (await client.call_tool("list_hardware", {})).structured_content
        submitted = await client.call_tool(
            "submit_job", {"name": "mcp", "script": "print('over mcp')"}
        )
        job_id = submitted.structured_content["job_id"]
        wait_for(service, job_id, FINISHED)
        status = (await client.call_tool("job_status", {"job_id": job_id})).structured_content
        logs = (await client.call_tool("job_logs", {"job_id": job_id})).structured_content
        listed = (await client.call_tool("list_jobs", {})).structured_content
        missing = await client.call_tool("job_status", {"job_id": "nope"})
    assert hardware["backends"][0]["gpus"][0]["name"] == "Test GPU"
    assert status["state"] == "succeeded" and status["log_tail"] == ["over mcp"]
    assert status["workdir"] == str(service.store.folder(job_id))
    assert logs["lines"] == ["over mcp"]
    assert [j["job_id"] for j in listed["jobs"]] == [job_id]
    assert missing.is_error and "does not exist" in missing.content[0].text
