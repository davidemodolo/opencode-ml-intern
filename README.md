# opencode-ml-intern

An ML engineering agent for [opencode](https://opencode.ai) V2, inspired by Hugging Face's [ml-intern](https://github.com/huggingface/ml-intern). Ask it for a fine-tune: it reads papers, docs and datasets, writes the training script, runs it as a job on your GPU and picks the work up again when the job ends.

![ML Intern reporting a LoRA smoke run on the local GPU](docs/images/lora-result.png)

It is made of two packages:

| Package | What it is |
|---|---|
| [`plugin/`](plugin/README.md) | The opencode plugin `opencode-ml-intern`: agents, prompt, skills, compute gate, loop guard and job notifications |
| [`server/`](server/README.md) | The MCP server `ml-intern-mcp`: research tools (papers, HF docs, datasets, GitHub) and GPU compute tools |

## Install

```sh
opencode plugin add opencode-ml-intern
```

Requirements: opencode 2.3.0, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU for training jobs. The plugin registers the MCP server itself. Select the `ml-intern` agent (Tab in the TUI) and describe the training you want.

## How it works

```mermaid
flowchart LR
    user([You]) --> tui[opencode TUI]
    subgraph opencode[opencode server]
        tui --> agent[ml-intern agent]
        agent -->|delegates crawls| research[ml-research subagent]
        plugin[[opencode-ml-intern plugin<br/>gate, loop guard, plan,<br/>session context]] -. hooks .- agent
    end
    subgraph mcp[ml-intern-mcp server]
        rtools[research tools]
        ctools[compute tools]
    end
    agent --> rtools
    research --> rtools
    agent -->|submit_job| ctools
    rtools --> web[(HF Hub, papers,<br/>docs, GitHub)]
    ctools -->|uv run --script| gpu[(local GPU)]
    gpu -->|events/*.json| plugin
    plugin -->|wakes the session| agent
```

1. The `ml-intern` agent delegates literature crawls to the `ml-research` subagent and checks datasets, models and current APIs with the research tools.
2. It writes a training script with inline PEP 723 dependencies and submits it with `submit_job`: a smoke run first, then the full run.
3. The server runs the job detached on the GPU, one job per GPU, and writes a completion event.
4. The plugin delivers the event to the session that submitted the job; the agent resumes, reads the logs and continues.

### A job, step by step

```mermaid
sequenceDiagram
    participant A as ml-intern agent
    participant P as plugin
    participant S as ml-intern-mcp
    participant G as GPU job
    A->>P: submit_job(script, timeout, vram)
    P->>P: approval and session limits
    P->>S: submit_job
    S->>G: uv run --script (detached)
    S-->>A: job id, "completion will be notified"
    Note over A: the turn ends, the GPU works
    G->>S: exit code
    S->>S: writes events/ID.json
    S-->>P: event
    P->>A: job outcome and last log lines (session resumes)
    A->>S: job_logs, next step
```

Jobs keep running when opencode or the MCP server restarts, and the event is delivered when the session comes back.

## What it looks like

The agent checks the hardware and submits a job:

![Hardware table and job submission](docs/images/gpu-check-submit.png)

When the job ends the session wakes up by itself with the outcome:

![Job completion notification and GPU report](docs/images/gpu-check-result.png)

## Try it from this repository

`e2e/` is an opencode project wired to the local plugin and server, with its own config and data folders:

```sh
nvm use                                     # Node 26 from .nvmrc
(cd plugin && npm ci && npm run build)
cd e2e
XDG_CONFIG_HOME=$PWD/.config ML_INTERN_DATA_DIR=$PWD/.data opencode --standalone
```

Pick the `ml-intern` agent with Tab and try, from the quickest to the longest:

| Prompt | What it exercises |
|---|---|
| `List the available hardware` | GPU detection and free VRAM |
| `Submit a job that prints the GPU name and the CUDA version with torch, 5 minute timeout` | Submit, detached run, completion notice and resume |
| `Fine-tune Qwen/Qwen2.5-0.5B-Instruct with LoRA on trl-lib/Capybara, 10 steps, no Hub push` | Research, script writing, training and adapter saved |
| `/jobs` | Job summary of the session |

Jobs live in `e2e/.data/jobs/<id>/` (`script.py`, `stdout.log`, `job.json`, `outputs/`), so `tail -f e2e/.data/jobs/<id>/stdout.log` and `nvidia-smi` show a job while it runs. `e2e/opencode.jsonc` sets `autoApprove: true`; set it to `false` to get the approval prompt on every submission.

## Development

Requirements: Node.js 26 (see `.nvmrc`), Python 3.12+ and uv.

```sh
cd server && uv sync && uv run pytest
cd plugin && npm install && npm run build && npm test
node e2e/run.mjs --fresh   # end-to-end acceptance run on the local GPU
```

`node e2e/run.mjs` starts an isolated `opencode serve`, asks for a 10-step LoRA fine-tune and exits with 0 once an adapter is saved. `spikes/` holds the probes used to verify opencode behavior (results in `docs/spikes.md`). The roadmap is in `docs/plan-phase-1.md` (local GPU) and `docs/plan-phase-2.md` (cloud GPU providers and HF Jobs).

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
