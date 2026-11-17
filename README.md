# opencode-ml-intern

An ML engineering agent for [opencode](https://opencode.ai) V2, inspired by Hugging Face's [ml-intern](https://github.com/huggingface/ml-intern). It is made of two packages:

- `plugin/`: the opencode plugin `opencode-ml-intern` (agents, prompt, compute gate, loop guard, job notifications).
- `server/`: the MCP server `ml-intern-mcp` (research tools and GPU compute tools).

Phase 1 runs jobs on the local GPU. Phase 2 adds cloud GPU providers and HF Jobs. See `docs/plan-phase-1.md` and `docs/plan-phase-2.md`.

## Requirements

- opencode 2.3.0
- Node.js 26 (see `.nvmrc`)
- Python 3.12+ and [uv](https://docs.astral.sh/uv/)

## Development

```sh
cd server && uv sync && uv run pytest
cd plugin && npm install && npm run build && npm test
cd e2e && opencode run --standalone "Call the ml-intern MCP server_info tool"
```

The `e2e/` project loads the plugin from `plugin/` and starts the server from `server/`.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
