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
- Skills: `ml-job-script`, `lora-sft`, `qlora`, `preference-tuning`.
- Compute gate: `submit_job` asks for approval and each session has a limit of active jobs and GPU hours.
- Job notifications: when a job ends, the session that submitted it receives its state and last log lines and resumes.
- Repetition guard: repeated identical tool calls and repeating call cycles are blocked, and the agent receives a corrective message.

## Options

```jsonc
{
  "plugins": [
    {
      "package": "opencode-ml-intern",
      "options": {
        "maxActiveJobs": 4,      // active jobs per session
        "maxGpuHours": 24,       // GPU hours per session, active jobs count with their timeout
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
