# Phase 1 spikes (M0)

Run on opencode 2.3.0 with the probe project in `spikes/`. Each section states the question, what was observed and the decision for M3.

## S1. Waking an idle session

Question: does `session.synthetic({ resume: true })` wake a session that already went idle?

Observed: a tool scheduled a timer and returned; the session went idle (`experimental.idle` fired with no background work). Fifteen seconds later the plugin called `session.synthetic` with `resume: true`. The call returned a `synthetic` inbox entry with delivery `steer`, the session ran a new step and the model replied as instructed.

Constraint: the timer lives in the opencode server process. It works with the background service and with `opencode serve`; a `opencode run --standalone` process exits when the session goes idle.

Decision: the plugin wakes sessions with `synthetic({ resume: true })` when a job completion event appears. Completion events are persisted by the MCP server, and on plugin setup any undelivered event is delivered, so a server restart loses no notification.

## S2. Permission requests raised inside `execute`

Question: how does the TUI show a permission request raised by a tool called from a code mode script?

Observed:

- The TUI shows "Permission required", "Call tool <action>" and "Tool: <action>", with Allow once and Reject. Approving runs the tool; rejecting returns `Tool.Error` to the model.
- Passing `save` to `permission.assert` adds an Always allow option.
- The `resources`, the `metadata` passed to `assert` and the `message` set in the `evaluate` hook reach the hook but are not rendered in the prompt.

Decision: the compute gate uses one action per backend (`mlintern_compute`) with `save: ["<backend>:*"]`, so a user can always allow local jobs. The cost and hardware of a job are stated by the model before submitting (prompt rule) and returned by `submit_job`. Automatic limits (concurrency, GPU hours, budget) are enforced by the `evaluate` hook with `deny`, which needs no rendering.

## S3. Plugin skills in subagents

Question: are skills added with `skill.transform` visible to a subagent?

Observed: with `permissions` `* deny`, `read allow` and `skill allow`, the subagent sees the plugin skill in its context and loads it through the `skill` tool (action `skill`, resource = skill id). Without the `skill` rule the skill is hidden.

Decision: `ml-research` gets `skill allow` (optionally scoped to the plugin skill ids).

## S4. Blocking a single repeated MCP call

Question: can a hook block one repeated MCP tool call?

Observed: both mechanisms work on a call made from inside `execute`. The first call succeeds; later identical calls fail with the guard message and the model reads that message:

- throwing in `tool.hook("execute.before")`;
- setting `effect = "deny"` with a `message` in `permission.hook("evaluate")`.

Decision: the loop guard uses `execute.before`, which receives the tool input needed for the call hash. `evaluate` stays dedicated to the compute gate.

## Notes

- The user's global opencode configuration (plugins, skills) also loads in the test projects. Integration tests run with an isolated config directory.
- MCP tools of a server named `ml-intern` are called in code mode as `tools["ml-intern"].<tool>()` and appear in permissions as `ml-intern_<tool>`.
