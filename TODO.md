# TODO

Phase 2 of [`docs/plan-phase-2.md`](docs/plan-phase-2.md): a single `ml-intern-mcp` server that runs jobs on the local GPU, GPU clouds and HF Jobs with the same tool contract, with spending controlled in dollars.

## Before phase 2

- [ ] Publish `ml-intern-mcp` on PyPI and `opencode-ml-intern` on npm, so `opencode plugin add opencode-ml-intern` works outside this repository
- [ ] Show "ML Intern" in the TUI: opencode 2.3.0 labels agents from their id, and the agent `name` is already set

## Multi-GPU jobs

- [x] `submit_job` takes `gpus: N`; the scheduler reserves N free GPUs together and sets `CUDA_VISIBLE_DEVICES` to all of them
- [x] VRAM check on the sum of the reserved GPUs
- [x] Plugin: GPU hours count `gpus x hours`; default session limits derived from the detected hardware
- [x] Skill for multi-GPU training: `torchrun` or `accelerate` from a PEP 723 script, DDP, FSDP, `device_map="auto"`
- [ ] Run a DDP smoke job on a machine with two GPUs

## M5. Backend abstraction

- [x] `Backend` interface with `hardware()`, `submit()`, `status()`, `logs()`, `cancel()`, `price()`; `local` implements it
- [x] Server config file with the enabled backends and their options; credentials stay with `hf auth login` and `sky check`
- [x] `list_hardware` merges the catalogs of all backends
- [x] Poller for remote jobs that writes their state to the data folder and the `events/` the plugin watches
- [x] On restart the poller picks up remote jobs from the saved state

## M6. SkyPilot backend

- [x] `skypilot==0.14.3` in its own uv environment (its `click` pin conflicts with `huggingface_hub`), managed jobs for recovery on preemption
- [x] `submit_job` mapped to a SkyPilot task: resources, `uv` setup, script and timeout
- [ ] Dataset and output mounts (outputs go to the Hub for now)
- [x] Price and availability catalog in `list_hardware`
- [x] Enabled providers (`sky check`) in the `list_hardware` status
- [ ] Short test job on a cheap provider (RunPod or Vast)
- [ ] Load SkyPilot in the server process instead of through `sky_bridge.py` once `skypilot` allows `click>=8.4.2` (0.14.3 pins `click<8.2`, `huggingface-hub==2.4.0` needs `click>=8.4.2`; the SDK calls the backend uses work with `click==8.5.1` forced)

## M7. HF Jobs backend

- [x] Adapter on `huggingface_hub` (`run_uv_job`, `inspect_job`, `fetch_job_logs`, `cancel_job`)
- [x] HF flavor and price catalog
- [x] Same `submit_job` contract, with PEP 723 script
- [ ] Short test job on a CPU flavor

## M8. Dollar gate

- [x] Estimated cost: hourly price times job timeout, checked in `execute.before` on `submit_job`
- [x] Per-session and global budget in `ctx.storage`: allow below the threshold, ask near the limit, deny above it
- [x] Reconciliation with the actual spend when a job ends; remaining budget in the session context
- [x] Autonomous mode: allow up to the budget, then deny
- [ ] Check in the TUI the approval raised by `permission.assert` from `execute.before` near the budget

## M9. Remote sandbox

- [ ] Spike: `shell.hook("create.before")` runs the native shell over ssh on the job machine
- [ ] If the spike holds, the agent works on the sandbox with the native tools
- [ ] Otherwise `sandbox_create`, `sandbox_bash`, `sandbox_read` and `sandbox_write` on the server, with prompt guidance for remote backends

## M10. Traces (optional)

- [ ] Collect sessions from `ctx.event.subscribe` and convert them to ml-intern's message format
- [ ] Upload to an HF dataset chosen by the user, opt-in in the config

## Done when

- [ ] The phase 1 scenario runs on `local`, on a provider through SkyPilot and on HF Jobs, switching only the backend
- [ ] The gate blocks a job over budget and the reconciled spend matches the provider's
- [ ] A remote job interrupted by preemption resumes and the session receives the completion notice
