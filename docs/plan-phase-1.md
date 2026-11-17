# Phase 1: everything working on the local GPU

Goal: reproduce ml-intern inside opencode V2 with an MCP server and a plugin, using the local machine's GPU as the only compute backend. The compute tool contract is backend agnostic, so phase 2 adds providers without touching the plugin or the prompt.

## Repository layout

```
opencode-ml-intern/
  plugin/   npm package `opencode-ml-intern` (TypeScript, compiled to JS)
  server/   PyPI package `ml-intern-mcp` (Python, stdio MCP server)
  docs/
  LICENSE   Apache-2.0
  NOTICE    attribution for code ported from huggingface/ml-intern
```

Starting versions, pinned with `==`: `@opencode/plugin==2.3.0`, `@opencode/cli==2.3.0` for tests, `typescript==7.0.2`, `mcp==2.3.0`, `huggingface-hub==2.4.0`, `datasets==5.2.0`, `nvidia-ml-py==13.615.71`. Training libraries (`torch`, `transformers`, `trl`, `peft`) live in the jobs, not in the server.

## Verified behavior of opencode 2.3.0

The plan relies on these behaviors, confirmed with probe plugins:

- `tool.transform` registers tools; `tool.hook("execute.before" | "execute.after")` receives `tool`, `sessionID`, `agent`, `input`, and `result` or `error`.
- `permission.assert` inside a tool requests permission; `permission.hook("evaluate")` can set `effect` to `allow`, `ask` or `deny`.
- `agent.transform` with `update` on a new id creates the agent; an `.md` file with the same id takes precedence.
- `session.synthetic` injects a message that the model acts on.
- `session.instructions.entry.put` adds text to the system prompt, subagents included.
- `session.hook("compaction")` fires and accepts a summary from the plugin.
- `skill.transform` adds skills visible to the primary agent.
- MCP tools appear in permissions as `<server>_<tool>` and in code mode as `tools.<server>.<tool>()`.
- A subagent sees only the tools it is allowed; delegation is the `subagent` action with the agent name as resource.
- `ctx.storage` offers `get`, `set`, `remove`, `scan`; `ctx.event.subscribe` receives every session event.
- There is no native doom loop detection.
- In code mode the model can poll in a loop inside a single `execute` script.

## M0. Scaffold and spikes

Monorepo scaffold, test runners (`vitest` for the plugin, `pytest` for the server), an opencode test project in `e2e/` that loads the plugin and the server from the repo.

Spikes, closed with the probe in `spikes/` (details in `spikes.md`):

| Spike | Question | Outcome |
|---|---|---|
| S1 | Does `session.synthetic({resume: true})` wake an idle session? | Yes, while the opencode server process stays alive |
| S2 | How does the TUI show a permission request raised inside `execute`? | Generic prompt with the action name; `save` adds Always allow; resources and messages are not shown |
| S3 | Are plugin skills visible to a subagent? | Yes, with an explicit `skill` allow rule |
| S4 | Can a hook block a single repeated MCP call? | Yes, both from `execute.before` and from `evaluate` |

## M1. MCP server: research tools

Ported from ml-intern and adapted to the MCP server:

- `hf_papers`: paper search and section reading.
- `hf_docs_explore`, `hf_docs_fetch`, `hf_api_find`: HF documentation.
- `hf_inspect_dataset`: schema, splits, samples, statistics.
- `hf_repo_files`: list and read files of model, dataset and Space repos.
- `github_find_examples`, `github_list_repos`, `github_read_file`: code examples.

On-disk cache of HTTP responses keyed by request, with a `--fresh` option to bypass it. Tests use recorded responses.

## M2. MCP server: local compute

Tools, identical in both phases:

| Tool | What it does |
|---|---|
| `list_hardware` | Available GPUs with free VRAM, backend and hourly cost (0 locally) |
| `submit_job` | Launches a script with hardware, timeout and env vars; returns the id |
| `job_status` | State, duration, exit code, last log lines |
| `job_logs` | Full logs or a window |
| `cancel_job` | Stops the job |
| `list_jobs` | Jobs of the session or of all sessions |

`local` backend:

- Each job has a folder in `~/.local/share/ml-intern-mcp/jobs/<id>/` with the script, `status.json`, `stdout.log` and the exit code.
- The script is a Python file with inline dependencies (PEP 723) run with `uv run`, so every job has its own environment. Docker stays an option for images with specific CUDA builds.
- The process starts with `setsid` and a wrapper that writes the exit code: it survives a server restart, and on restart the server rebuilds state from the folders.
- One GPU runs one job at a time; the others are queued.
- Before launch `nvidia-ml-py` checks free VRAM; if the job declares a higher requirement, `submit_job` fails immediately with the reason.
- `job_status` on a running job answers that completion will be notified, to discourage polling.
- A job completion event is written to `events/` in the data folder, which the plugin watches.

Tests: a fake job that prints and exits, a failing job, timeout, cancel, server restart with a running job.

## M3. opencode plugin

- **Agents**: `ml-intern` (primary) and `ml-research` (read-only subagent: `* deny`, `read`, `skill` and the research MCP tools `allow`), created with `agent.transform`.
- **Prompt**: port of `system_prompt_v3` adapted to code mode, MCP tools and the local GPU. Dynamic parts (hardware, active jobs, limits) go in `instructions.entry`, refreshed on every `prompt`.
- **Compute gate**: `submit_job` goes through an `mlintern_compute` permission with resource `<backend>:<gpu>:<hours>` and `save: ["<backend>:*"]`. The model states hardware and cost before submitting, since the TUI prompt shows only the action. Locally `evaluate` limits concurrent jobs and GPU hours per session, stored in `ctx.storage`; in phase 2 the same point counts dollars.
- **Doom loop**: hash of tool, normalized input and result on `execute.after`. At the threshold the call is blocked by throwing in `execute.before` and a corrective `synthetic` message is sent.
- **Wake up**: watcher on the `events/` folder; when a job ends, `synthetic` with `resume: true`, the outcome and the last log lines. Undelivered events are delivered on plugin setup.
- **`plan` tool**: list of steps with status, stored in `ctx.storage`, shown in the instructions.
- **`/jobs` command**: job summary.
- **Skills**: TRL recipes (SFT, DPO, GRPO), PEFT and LoRA, quantization for 8 GB, kernel selection.
- **Compaction**: the hook keeps the plan and the active jobs in the summary.

Unit tests of the hooks with a fake context; integration tests in the `e2e/` project with `opencode run --standalone`.

## M4. End to end, packaging and docs

- Acceptance scenario: "LoRA fine-tune of a 0.5B model on an HF dataset and publish the adapter", run by `ml-intern` end to end on the local GPU.
- npm package for the plugin and PyPI package for the server; one install step that adds the plugin and the MCP server to the opencode config.
- README with installation, configuration and hardware limits.
- CI with lint, tests and the compatibility probes against the pinned opencode version.

## Definition of done

- Every MCP tool has passing tests.
- The acceptance scenario completes with no manual intervention beyond approvals.
- A job survives a server restart and the session receives the completion notice.
- Repeated polling is stopped by the guard.
