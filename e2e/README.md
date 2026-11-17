# End-to-end project

An opencode project that loads the plugin from `../plugin` and starts the MCP server from `../server`. It runs with an isolated config directory (`.config/`) and data directory (`.data/`), so the user's global opencode configuration and jobs are left untouched.

## Acceptance run

```sh
cd plugin && npm run build && cd ..
node e2e/run.mjs           # reuse .data from previous runs
node e2e/run.mjs --fresh   # start from an empty data directory
```

The script starts `opencode serve`, asks the `ml-intern` agent for a 10-step LoRA fine-tune of Qwen2.5-0.5B on the local GPU and waits until a job has saved an adapter and no job is running. It exits with 0 on success.

Environment variables: `E2E_MODEL` (default `opencode/deepseek-v4.1-flash-free`), `E2E_PORT` (default 4799).
