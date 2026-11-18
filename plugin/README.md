# opencode-ml-intern

An [opencode](https://opencode.ai) plugin that adds ML Intern, an ML engineering agent inspired by Hugging Face's [ml-intern](https://github.com/huggingface/ml-intern): it researches training recipes in papers, docs and code, writes training scripts and runs them as GPU jobs.

## Install

Requirements: opencode 2.3.0, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU for training jobs (jobs also run on CPU).

```sh
opencode plugin add opencode-ml-intern
```

The plugin registers the `ml-intern` MCP server (`uvx ml-intern-mcp==0.1.0`) unless a server with that name is already configured. Select the `ml-intern` agent in opencode to start.

## What it adds

- Agents: `ml-intern` (primary) and `ml-research` (read-only research subagent for literature crawls, dataset checks and code examples).
- Tool `plan_tool`: a todo list shown to the agent in every request.
- Command `/jobs`: lists the jobs of the session.
- Skills: `ml-job-script`, `lora-sft`, `qlora`, `preference-tuning`, `multi-gpu`.
- Compute gate: `submit_job` asks for approval and each session has a limit of active jobs and of local GPU hours (GPUs times hours).
- Dollar budget: a job on a paid backend (`hf_jobs`, `skypilot`) reserves its hourly price times its timeout. It runs while the session stays under `maxUsd` (and all sessions under `maxUsdTotal`, when set), asks for approval once the spend would pass `budgetAskRatio` of a budget, and is refused above it. When the job ends its reservation is replaced by the actual cost. In autonomous mode there is no approval step: jobs run up to the budget.
- Hub writes: `hf_repo_write` (uploads and deletes in Hugging Face repos) asks for approval on every call.
- Job notifications: when a job ends, the session that submitted it receives its state and last log lines and resumes.
- Repetition guard: repeated identical tool calls and repeating call cycles are blocked, and the agent receives a corrective message. Polling `job_status` on a running job is blocked after `loopThreshold` calls, since completion is notified.

## Options

```jsonc
{
  "plugins": [
    {
      "package": "opencode-ml-intern",
      "options": {
        "maxActiveJobs": 4,      // active jobs per session
        "maxGpuHours": 24,       // local GPU hours per session, active jobs count with their timeout; default 24 per local GPU
        "maxUsd": 0,             // dollars per session on paid backends; 0 refuses paid jobs
        "maxUsdTotal": 100,      // dollars across all sessions on paid backends; unset means no total limit
        "budgetAskRatio": 0.8,   // share of a budget above which a paid job asks for approval
        "autoApprove": false,    // true submits jobs without asking
        "autonomous": false,     // true adds the headless autonomous-mode rules to the prompt
        "loopThreshold": 3,      // identical calls before the repetition guard triggers
        "mcpServer": "ml-intern",
        "serverCommand": ["uvx", "ml-intern-mcp==0.1.0"],
        "dataDir": "~/.local/share/ml-intern-mcp"
      }
    }
  ]
}
```

Job notifications need a long-lived opencode server (the background service or `opencode serve`).

Remote backends are enabled in the MCP server's config file, see [`server/README.md`](../server/README.md#config-file).
