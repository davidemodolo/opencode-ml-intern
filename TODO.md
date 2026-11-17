# TODO

Phase 2 of [`docs/plan-phase-2.md`](docs/plan-phase-2.md): a single `ml-intern-mcp` server that runs jobs on the local GPU, GPU clouds and HF Jobs with the same tool contract, with spending controlled in dollars.

## Before phase 2

- [ ] Publish `ml-intern-mcp` on PyPI and `opencode-ml-intern` on npm, so `opencode plugin add opencode-ml-intern` works outside this repository
- [ ] `submit_job`: accept the path of an existing `.py` file in `script` as well as the script source
- [ ] Show "ML Intern" in the TUI: opencode 2.3.0 labels agents from their id, and the agent `name` is already set

## Multi-GPU jobs

- [ ] `submit_job` takes `gpus: N`; the scheduler reserves N free GPUs together and sets `CUDA_VISIBLE_DEVICES` to all of them
- [ ] VRAM check on the sum of the reserved GPUs
- [ ] Plugin: GPU hours count `gpus x hours`; default session limits derived from the detected hardware
- [ ] Skill for multi-GPU training: `torchrun` or `accelerate` from a PEP 723 script, DDP, FSDP, `device_map="auto"`

## M5. Backend abstraction

- [ ] `Backend` interface with `hardware()`, `submit()`, `status()`, `logs()`, `cancel()`, `price()`; `local` implements it
- [ ] Server config file with the enabled backends and their credentials
- [ ] `list_hardware` merges the catalogs of all backends
- [ ] Poller for remote jobs that writes their state to the data folder and the `events/` the plugin watches
- [ ] On restart the poller picks up remote jobs from the saved state

## M6. SkyPilot backend

- [ ] `skypilot==0.14.3` as a library, managed jobs for recovery on preemption
- [ ] `submit_job` mapped to a SkyPilot task: resources, image, `uv` setup, dataset and output mounts
- [ ] Price and availability catalog in `list_hardware`
- [ ] `sky check` at startup, credential state in `list_hardware`
- [ ] Short test job on a cheap provider (RunPod or Vast)

## M7. HF Jobs backend

- [ ] Adapter on `huggingface_hub` (`run_job`, `inspect_job`, `fetch_job_logs`, `cancel_job`)
- [ ] HF flavor and price catalog
- [ ] Same `submit_job` contract, with image and PEP 723 script

## M8. Dollar gate

- [ ] Estimated cost: hourly price times job timeout, checked in `execute.before` on `submit_job`
- [ ] Per-session and global budget in `ctx.storage`: allow below the threshold, ask near the limit, deny above it
- [ ] Reconciliation with the actual spend when a job ends; remaining budget in the session context
- [ ] Autonomous mode: allow up to the budget, then deny

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
