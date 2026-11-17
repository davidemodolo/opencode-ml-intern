# Phase 2: single multi-provider MCP server

Goal: the same `ml-intern-mcp` server runs jobs on the local GPU, GPU clouds and HF Jobs with the phase 1 tool contract, and the plugin controls spending in dollars.

## Landscape (November 2026)

- RunPod has an official MCP server limited to RunPod.
- Across providers there are only small community projects (infra-mcp, SkyMCP, Train in Silence) and a paid router (FastGPU).
- SkyPilot and dstack unify dozens of clouds, Kubernetes and SSH nodes, with no official MCP server.

Decision: a single server with pluggable backends, where SkyPilot covers cloud providers and HF Jobs has a dedicated adapter.

## M5. Backend abstraction

- `Backend` interface with `hardware()`, `submit()`, `status()`, `logs()`, `cancel()`, `price()`, already used by the phase 1 `local` backend.
- Enabled backends and credentials configured in the server config file; `list_hardware` merges the catalogs.
- Remote job state in the same data folder as phase 1, updated by a poller in the server; the plugin keeps watching `events/`.
- Resumable poller: on restart it picks up remote jobs from the saved state.

## M6. SkyPilot backend

- `skypilot==0.14.3` as a library, with managed jobs (`sky jobs launch`) for recovery on preemption.
- `submit_job` maps to a SkyPilot task: resources, image, setup with `uv`, dataset and output mounts.
- Price and availability catalog from SkyPilot for `list_hardware`.
- Credential check with `sky check` at startup, exposed in `list_hardware`.
- Test on a cheap provider (RunPod or Vast) with a short job.

## M7. HF Jobs backend

- Adapter on `huggingface_hub` (`run_job`, `inspect_job`, `fetch_job_logs`, `cancel_job`).
- HF flavor and price catalog.
- Same `submit_job` contract, with image and PEP 723 script.

## M8. Dollar gate

- Estimated cost: hourly price times job timeout, passed as resource in the `mlintern_compute` `assert`.
- Per-session and global budget in `ctx.storage`; `evaluate` returns `allow` below the threshold, `ask` near the limit, `deny` above it.
- Reconciliation with actual spend when the job ends; remaining budget shown in the instructions.
- Automatic mode equivalent to ml-intern's "yolo": `allow` up to the budget, then `deny`.

## M9. Remote sandbox

- Spike: `shell.hook("create.before")` redirects the native shell over ssh to the job machine.
- If the spike holds, the agent works on the sandbox with the native tools.
- Otherwise the server exposes `sandbox_create`, `sandbox_bash`, `sandbox_read`, `sandbox_write` and the prompt steers the model to them when the backend is remote.

## M10. Traces (optional)

- Collection from `ctx.event.subscribe` and conversion to ml-intern's message format.
- Upload to an HF dataset chosen by the user, with explicit opt-in in the config.

## Definition of done

- The phase 1 scenario runs unchanged on `local`, on a provider through SkyPilot and on HF Jobs, switching only the backend.
- The gate blocks a job over budget and the reconciled spend matches the provider's.
- A remote job interrupted by preemption resumes and the session receives the completion notice.
