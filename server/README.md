# ml-intern-mcp

MCP server with machine learning research tools and GPU compute tools, used by the `opencode-ml-intern` plugin and usable from any MCP client.

## Research tools

Ported from [ml-intern](https://github.com/huggingface/ml-intern) (Apache-2.0):

| Tool | What it does |
|---|---|
| `hf_papers` | Searches papers, reads sections, finds linked models, datasets and collections, citation graph |
| `explore_hf_docs` | Lists the pages of an HF library documentation |
| `fetch_hf_docs` | Reads an HF documentation page as markdown |
| `find_hf_api` | Finds HF Hub REST endpoints with curl examples |
| `hf_inspect_dataset` | Shows dataset configs, splits, schema and sample rows |
| `hf_repo_files` | Lists and reads files in HF repos |
| `hf_repo_write` | Uploads and deletes files in HF repos |
| `github_find_examples` | Finds example files in a GitHub repository |
| `github_list_repos` | Lists the repositories of a GitHub user or organization |
| `github_read_file` | Reads a file from a GitHub repository |

Tokens: `HF_TOKEN` or the cached `hf auth login` token; `GITHUB_TOKEN` or the `gh auth token` of the GitHub CLI.

## Compute tools

| Tool | What it does |
|---|---|
| `list_hardware` | Backends with their local GPUs (total and free VRAM) or remote hardware options (GPU, count, VRAM, USD per hour), cheapest first, and the backend status; `gpu` filters the options |
| `submit_job` | Runs a Python script with inline PEP 723 dependencies as a background job on a backend; takes the script source or the absolute path of a `.py` file |
| `job_status` | State, exit code, duration, cost and last log lines |
| `job_logs` | The last lines of the log, or a window from an offset |
| `cancel_job` | Stops a queued or running job |
| `list_jobs` | Jobs, optionally filtered by state |

Every backend takes the same `submit_job` call: a PEP 723 script run with `uv run --script`, a timeout and environment variables. `backend` picks where it runs and `hardware` picks a remote machine type from `list_hardware`.

### Backends

| Backend | Where jobs run | Enabled by |
|---|---|---|
| `local` | This machine's GPUs | Always |
| `hf_jobs` | [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs), on the flavor given as `hardware` (`t4-small`, `a10g-large`, `a100x8`...) | `[backends.hf_jobs]` in the config file and an HF token |
| `skypilot` | The cheapest matching machine across the providers set up in [SkyPilot](https://docs.skypilot.co), as a managed job that is relaunched after a preemption; `hardware` is an accelerator and count such as `A100:1` | `[backends.skypilot]` in the config file and `sky check` passing for at least one provider |

`local` runs each job in its own process group, detached from the server. `gpus` sets how many GPUs a job gets together (`CUDA_VISIBLE_DEVICES` lists them, default 1); a job waits in the queue until that many GPUs are free (one job at a time on machines without a GPU). A job that asks for more GPUs than the machine has, or more `vram_gb` than its largest GPUs have together, is rejected at submit time.

Remote jobs are handed to their provider at submit time and polled every `poll_seconds`; `job_status` and `job_logs` poll at once. When a remote job ends the server saves its log and its cost, the hourly price times the running time. The working directory of a remote job is on the provider's machine and is deleted at the end, so scripts push their outputs to the Hub. `hf_jobs` and `skypilot` pass the local HF token to the job as the `HF_TOKEN` secret.

SkyPilot runs in its own environment, since its dependencies conflict with this server's: the server calls it through `uv run --with skypilot[<providers>]==0.14.3`, which installs it on first use, and SkyPilot starts its local API server. Managed jobs also start SkyPilot's jobs controller on one of the configured providers. The SkyPilot catalog reports device memory per GPU for some providers and per machine for others, so `vram_gb` is shown but not checked on `skypilot`.

### Config file

`$XDG_CONFIG_HOME/ml-intern-mcp/config.toml` (or the path in `ML_INTERN_CONFIG`) enables the remote backends; each section is optional:

```toml
[backends.hf_jobs]
namespace = "my-org"        # bill jobs to an organization; default: the token's user
poll_seconds = 30

[backends.skypilot]
infra = ["runpod", "vast"]  # providers to use (cloud or cloud/region); default: every enabled one
use_spot = false            # spot instances, cheaper and preemptible
max_restarts_on_errors = 0  # relaunches after a failure of the script itself
poll_seconds = 60
```

Credentials stay with each tool: `hf auth login` or `HF_TOKEN` for `hf_jobs`, the provider credentials checked by `sky check` for `skypilot`. `list_hardware` shows the status of each backend.

### Job files

Each job lives in `$XDG_DATA_HOME/ml-intern-mcp/jobs/<id>/` (`job.json`, `script.py`, `stdout.log`, `exit_code`), so local jobs keep running and remote jobs keep being polled when the server restarts. When a job ends the server writes `events/<id>.json` with its state, exit code, duration, cost and last log lines; the opencode plugin reads these events to notify the session and removes each one once delivered. `list_hardware` also writes `catalog.json` with the local GPU count and the hourly price of every remote option, which the plugin reads to check budgets.

Environment variables: `ML_INTERN_DATA_DIR` overrides the data folder, `ML_INTERN_CONFIG` the config file, `ML_INTERN_JOB_RUNNER` the local job command (default `uv run --script`).

## Running

```sh
uv run ml-intern-mcp            # stdio server
uv run ml-intern-mcp --fresh    # clear the result cache first
uv run ml-intern-mcp --no-cache # disable the result cache
```

Successful results of read-only tools are cached on disk for 6 hours in `$XDG_CACHE_HOME/ml-intern-mcp`.

## Development

```sh
uv run pytest
ML_INTERN_LIVE=1 uv run pytest tests/test_research_live.py   # live calls to every research tool
```
