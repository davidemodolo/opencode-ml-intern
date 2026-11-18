"""Runs one SkyPilot SDK call in SkyPilot's own environment and writes the result as JSON.

Usage: python sky_bridge.py <request.json> <result.json>. Imports only sky and the standard library.
"""

import io
import json
import sys
from typing import Any

import sky

RECORD_FIELDS = ("job_id", "job_name", "status", "start_at", "end_at", "failure_reason")


def _value(item: Any) -> Any:
    return getattr(item, "value", item)


def _field(record: Any, name: str) -> Any:
    return record.get(name) if isinstance(record, dict) else getattr(record, name, None)


def enabled_infra() -> list[str]:
    return [str(cloud).lower() for cloud in sky.get(sky.enabled_clouds())]


def accelerators(clouds: list[str]) -> dict[str, list[dict[str, Any]]]:
    catalog = sky.list_accelerators(gpus_only=True, clouds=clouds or None)
    return {name: [item._asdict() for item in items] for name, items in catalog.items()}


def launch(config: dict[str, Any], name: str) -> str:
    return str(sky.jobs.launch(sky.Task.from_yaml_config(config), name=name))


def job(name: str) -> dict[str, Any] | None:
    records = [r for r in sky.get(sky.jobs.queue(refresh=False)) if _field(r, "job_name") == name]
    if not records:
        return None
    latest = max(records, key=lambda r: _field(r, "job_id"))
    return {key: _value(_field(latest, key)) for key in RECORD_FIELDS}


def request_error(request_id: str) -> str | None:
    for request in sky.api_status([request_id]):
        if str(_value(request.status)) == "FAILED":
            return str(request.error or "the launch request failed")
    return None


def logs(name: str) -> str:
    buffer = io.StringIO()
    sky.jobs.tail_logs(name=name, follow=False, output_stream=buffer)
    return buffer.getvalue()


def cancel(name: str) -> None:
    sky.get(sky.jobs.cancel(name=name))


COMMANDS = {
    "enabled_infra": enabled_infra,
    "accelerators": accelerators,
    "launch": launch,
    "job": job,
    "request_error": request_error,
    "logs": logs,
    "cancel": cancel,
}


def main() -> None:
    request_path, result_path = sys.argv[1:3]
    with open(request_path) as file:
        request = json.load(file)
    result = COMMANDS[request["command"]](*request["args"])
    with open(result_path, "w") as file:
        json.dump({"result": result}, file)


if __name__ == "__main__":
    main()
