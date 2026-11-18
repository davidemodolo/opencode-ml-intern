"""A stand-in for the SkyPilot SDK used by sky_bridge.py in tests; it records calls to calls.jsonl."""

import json
import os
from collections import namedtuple
from types import SimpleNamespace

Info = namedtuple(
    "Info",
    "cloud instance_type accelerator_name accelerator_count cpu_count device_memory memory "
    "price spot_price region",
)
LOG = os.environ["FAKE_SKY_LOG"]


def _record(name, *args):
    with open(LOG, "a") as file:
        file.write(json.dumps([name, *args], default=str) + "\n")


def get(request):
    return request


def enabled_clouds():
    return ["RunPod", "Vast"]


def list_accelerators(gpus_only, clouds):
    _record("list_accelerators", clouds)
    return {"A100": [Info("RunPod", "x", "A100", 1, 8, 80.0, 64, 1.6, float("nan"), "r")]}


def api_status(request_ids):
    return [SimpleNamespace(status=SimpleNamespace(value="FAILED"), error="no quota")]


class Task:
    def __init__(self, config):
        self.config = config

    @classmethod
    def from_yaml_config(cls, config):
        return cls(config)


class _Jobs:
    def launch(self, task, name):
        _record("launch", task.config, name)
        return "req-7"

    def queue(self, refresh):
        status = SimpleNamespace(value="RUNNING")
        return [
            {"job_id": 1, "job_name": "mli-a", "status": status, "start_at": 1.0, "end_at": None},
            {"job_id": 2, "job_name": "mli-a", "status": status, "start_at": 5.0, "end_at": None},
            {"job_id": 3, "job_name": "mli-b", "status": status, "start_at": 9.0, "end_at": None},
        ]

    def tail_logs(self, name, follow, output_stream):
        output_stream.write(f"log of {name}\n")

    def cancel(self, name):
        _record("cancel", name)


jobs = _Jobs()
