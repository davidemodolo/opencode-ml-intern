import json
import os
import sys
import threading
import time

import pytest
from mcp import Client

from ml_intern_mcp.compute.local import LocalBackend, _alive
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


def test_cancel_kills_the_script_process(tmp_path):
    service = make_service(tmp_path)
    script = (
        "import os, pathlib, time; pathlib.Path('pid').write_text(str(os.getpid())); time.sleep(60)"
    )
    job = service.submit(JobSpec(name="forever", script=script))
    pid_file = service.store.folder(job.id) / "pid"
    deadline = time.time() + 10
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.05)
    pid = int(pid_file.read_text())
    service.cancel(job.id)
    deadline = time.time() + 5
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)


def test_services_sharing_a_store_start_a_queued_job_once(tmp_path):
    services = [make_service(tmp_path) for _ in range(4)]
    for _ in range(5):
        job = services[0].store.create(JobSpec(name="once", script="open('runs', 'a').write('x')"))
        threads = [threading.Thread(target=s.reconcile) for s in services for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        job = wait_for(services[0], job.id, FINISHED)
        assert (services[0].store.folder(job.id) / "runs").read_text() == "x"


def test_timeout_exit_code_marks_the_job_timed_out(tmp_path):
    store = JobStore(tmp_path)
    backend = LocalBackend(store, [sys.executable], gpu_probe=lambda: list(ONE_GPU))
    job = store.create(JobSpec(name="slow", script="print(1)"))
    job.state = JobState.RUNNING
    store.exit_code_path(job.id).write_text("124")
    backend.refresh(job)
    assert job.state == JobState.TIMED_OUT and job.exit_code == 124


def test_sigkill_before_the_timeout_is_a_failure_not_a_timeout(tmp_path):
    service = make_service(tmp_path)
    script = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
    job = service.submit(JobSpec(name="oom", script=script))
    job = wait_for(service, job.id, FINISHED)
    assert job.state == JobState.FAILED and job.exit_code == 137
    assert "SIGKILL" in job.error


def test_sigkill_noticed_after_a_restart_uses_the_exit_time(tmp_path):
    store = JobStore(tmp_path)
    backend = LocalBackend(store, [sys.executable], gpu_probe=lambda: list(ONE_GPU))
    job = store.create(JobSpec(name="oom", script="print(1)", timeout_minutes=60))
    job.state = JobState.RUNNING
    job.started_at = time.time() - 2 * 3600
    exit_code = store.exit_code_path(job.id)
    exit_code.write_text("137")
    os.utime(exit_code, (job.started_at + 600, job.started_at + 600))
    backend.refresh(job)
    assert job.state == JobState.FAILED and job.duration_seconds == 600


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


async def test_submit_job_reads_a_script_path(tmp_path):
    service = make_service(tmp_path / "data")
    script = tmp_path / "train.py"
    script.write_text("print('from file')")
    async with Client(build_server([], service)) as client:
        submitted = await client.call_tool("submit_job", {"name": "file", "script": f" {script}\n"})
        job = wait_for(service, submitted.structured_content["job_id"], FINISHED)
        assert service.store.log_lines(job.id) == ["from file"]
        for bad in ["train.py", str(tmp_path / "missing.py")]:
            result = await client.call_tool("submit_job", {"name": "bad", "script": bad})
            assert result.is_error


async def test_log_and_list_windows_reject_empty_or_negative_sizes(tmp_path):
    service = make_service(tmp_path)
    async with Client(build_server([], service)) as client:
        submitted = await client.call_tool(
            "submit_job", {"name": "n", "script": "print(1); print(2)"}
        )
        job_id = submitted.structured_content["job_id"]
        wait_for(service, job_id, FINISHED)
        empty = await client.call_tool("job_logs", {"job_id": job_id, "tail": 0})
        bad = [
            await client.call_tool("job_logs", {"job_id": job_id, "offset": -1}),
            await client.call_tool("job_logs", {"job_id": job_id, "tail": -1}),
            await client.call_tool("job_logs", {"job_id": job_id, "offset": 0, "limit": 0}),
            await client.call_tool("list_jobs", {"limit": 0}),
        ]
    assert empty.structured_content["lines"] == []
    assert all(result.is_error for result in bad)


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
