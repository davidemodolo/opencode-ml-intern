# opencode-ml-intern

An ML engineering agent for [opencode](https://opencode.ai) V2, inspired by Hugging Face's [ml-intern](https://github.com/huggingface/ml-intern). It is made of two packages:

- `plugin/`: the opencode plugin `opencode-ml-intern` (agents, prompt, compute gate, loop guard, job notifications).
- `server/`: the MCP server `ml-intern-mcp` (research tools and GPU compute tools).

Phase 1 runs jobs on the local GPU. Phase 2 adds cloud GPU providers and HF Jobs. See `docs/plan-phase-1.md` and `docs/plan-phase-2.md`.

## Install

```sh
opencode plugin add opencode-ml-intern
```

Requirements: opencode 2.3.0, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU for training jobs. The plugin registers the MCP server itself; see `plugin/README.md` for options and `server/README.md` for the tools.

## How it works

1. The `ml-intern` agent delegates literature crawls to the `ml-research` subagent and checks datasets, models and current APIs with the research tools.
2. It writes a training script with inline PEP 723 dependencies and submits it with `submit_job` (smoke run first, then the full run).
3. The server runs the job detached on the GPU, one job per GPU, and writes a completion event.
4. The plugin delivers the event to the session that submitted the job; the agent resumes, reads the logs and continues.

## Development

Requirements: Node.js 26 (see `.nvmrc`), Python 3.12+ and uv.

```sh
cd server && uv sync && uv run pytest
cd plugin && npm install && npm run build && npm test
node e2e/run.mjs --fresh   # end-to-end acceptance run on the local GPU
```

`e2e/` is an isolated opencode project wired to the local plugin and server. `spikes/` holds the probes used to verify opencode behavior (results in `docs/spikes.md`).

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
