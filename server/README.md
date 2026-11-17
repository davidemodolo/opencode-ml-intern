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
| `hf_repo_files` | Lists, reads, uploads and deletes files in HF repos |
| `github_find_examples` | Finds example files in a GitHub repository |
| `github_list_repos` | Lists the repositories of a GitHub user or organization |
| `github_read_file` | Reads a file from a GitHub repository |

Tokens: `HF_TOKEN` or the cached `hf auth login` token; `GITHUB_TOKEN` or the `gh auth token` of the GitHub CLI.

## Compute tools

| Tool | What it does |
|---|---|
| `list_hardware` | Backends with their GPUs (total and free VRAM) and hourly cost |
| `submit_job` | Runs a Python script with inline PEP 723 dependencies as a background job |
| `job_status` | State, exit code, duration and last log lines |
| `job_logs` | The last lines of the log, or a window from an offset |
| `cancel_job` | Stops a queued or running job |
| `list_jobs` | Jobs, optionally filtered by state |

The `local` backend runs each job with `uv run --script` in its own process group, detached from the server, with `CUDA_VISIBLE_DEVICES` set to the assigned GPU. One job runs per GPU (one at a time on machines without a GPU) and the others wait in a queue. A job that declares more `vram_gb` than the largest GPU is rejected at submit time.

Each job lives in `$XDG_DATA_HOME/ml-intern-mcp/jobs/<id>/` (`job.json`, `script.py`, `stdout.log`, `exit_code`), so jobs keep running and are picked up again when the server restarts. When a job ends the server writes `events/<id>.json` with its state, exit code, duration and last log lines; the opencode plugin reads these events to notify the session.

Environment variables: `ML_INTERN_DATA_DIR` overrides the data folder, `ML_INTERN_JOB_RUNNER` overrides the job command (default `uv run --script`).

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
uv run python scripts/smoke_research.py   # live calls to every research tool
```
