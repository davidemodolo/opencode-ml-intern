import { Plugin } from "@opencode/plugin"
import { budgetAction, configureAgents, ML_INTERN_AGENT, ML_RESEARCH_AGENT } from "./agents.js"
import { Catalog } from "./catalog.js"
import { sessionContext } from "./context.js"
import { EventWatcher, formatEvent } from "./events.js"
import { hfUser } from "./hf.js"
import { registerServer } from "./mcp.js"
import { estimateUsd, limitsFor, requestOf, SubmitGate } from "./gate.js"
import { JobLedger, spentUsd } from "./ledger.js"
import { LoopGuard } from "./loop-guard.js"
import { PollGuard } from "./poll-guard.js"
import { resolveOptions } from "./options.js"
import { formatPlan, PLAN_TOOL_DESCRIPTION, PLAN_TOOL_INPUT, validatePlan, type PlanItem } from "./plan.js"
import { jobIdFrom } from "./results.js"
import { loadSkills } from "./skills.js"

const GUARDED_AGENTS = new Set([ML_INTERN_AGENT, ML_RESEARCH_AGENT])
const CODE_MODE_TOOL = "execute"
const COMPACTION_NOTE =
  "When summarizing this ML Intern session, keep the current plan, every job ID with its state, " +
  "the datasets and models in use and the results obtained so far."

const warn = (error: unknown) => console.warn("[opencode-ml-intern]", error)

export default Plugin.define({
  id: "ml-intern",
  async setup(ctx) {
    const options = resolveOptions(ctx.options ?? {}, process.env)
    const submitTool = `${options.mcpServer}_submit_job`
    const statusTool = `${options.mcpServer}_job_status`
    const ledger = new JobLedger(ctx.storage)
    const catalog = () => Catalog.read(options.dataDir)
    const gate = new SubmitGate(ledger, options, catalog)
    const guard = new LoopGuard(options.loopThreshold)
    const polls = new PollGuard(options.loopThreshold)
    const skills = await loadSkills()
    let user: string | undefined
    void hfUser(process.env).then((name) => (user = name))

    const planKey = (sessionID: string) => `plan/${sessionID}`
    const readPlan = async (sessionID: string) => ((await ctx.storage.get(planKey(sessionID))) ?? []) as unknown as PlanItem[]
    const notify = (sessionID: string, text: string, resume: boolean) =>
      ctx.session.synthetic({ sessionID: sessionID as any, text, resume }).then(() => undefined, warn)

    await ctx.mcp.transform((editor) => void registerServer(editor, options))
    await ctx.agent.transform((editor) => configureAgents(editor, options))

    await ctx.skill.transform((editor) => {
      for (const skill of skills) editor.add(skill as any)
    })

    await ctx.tool.transform((editor) => {
      editor.add({
        name: "plan_tool",
        description: PLAN_TOOL_DESCRIPTION,
        input: PLAN_TOOL_INPUT as any,
        execute: async (input: unknown, call: { sessionID: string }) => {
          const items = validatePlan(input)
          await ctx.storage.set(planKey(call.sessionID), items as any)
          return { content: formatPlan(items) } as any
        },
      } as any)
    })

    await ctx.command.transform((editor) => {
      editor.add({
        name: "jobs",
        description: "List the ML Intern jobs of this session",
        execute: async ({ sessionID }) => {
          await notify(sessionID, "Call list_jobs and show my jobs as a short table: job ID, name, state, duration.", true)
        },
      })
    })

    await ctx.tool.hook("execute.before", async (event) => {
      if (event.tool === CODE_MODE_TOOL) return
      if (GUARDED_AGENTS.has(event.agent)) {
        const reason =
          guard.check(event.sessionID, event.tool, event.input) ??
          (event.tool === statusTool ? polls.check(event.sessionID, event.input) : undefined)
        if (reason) throw new Error(reason)
      }
      if (event.tool === submitTool) {
        const request = requestOf(event.input)
        const decision = await gate.reserve(event.id, event.sessionID, request)
        if (decision.effect === "deny") throw new Error(decision.reason)
        if (decision.effect === "ask" && !options.autonomous) {
          await ctx.permission
            .assert({
              sessionID: event.sessionID,
              agent: event.agent,
              action: budgetAction(options),
              resources: [`${request.backend}:${request.hardware ?? ""}`],
              metadata: { description: decision.reason },
              source: { type: "tool", messageID: event.messageID, id: event.id },
            })
            .catch((error: unknown) => {
              gate.release(event.id)
              throw error
            })
        }
      }
    })

    await ctx.tool.hook("execute.after", async (event) => {
      if (event.tool === CODE_MODE_TOOL) return
      if (event.tool === submitTool) {
        const jobId = event.status === "completed" ? jobIdFrom(event.result) : undefined
        try {
          if (jobId) {
            const request = requestOf(event.input)
            await ledger.add({
              jobId,
              sessionID: event.sessionID,
              name: (event.input as { name?: string } | undefined)?.name ?? jobId,
              timeoutMinutes: request.timeoutMinutes,
              submittedAt: Date.now(),
              backend: request.backend,
              hardware: request.hardware,
              gpus: request.gpus,
              estimatedUsd: estimateUsd(request, await catalog()),
            })
          }
        } finally {
          gate.release(event.id)
        }
      }
      if (!GUARDED_AGENTS.has(event.agent)) return
      if (event.tool === statusTool && event.status === "completed") polls.record(event.sessionID, event.input, event.result)
      const outcome = event.status === "completed" ? event.result : event.error
      const correction = guard.record(event.sessionID, event.tool, event.input, outcome)
      if (correction) await notify(event.sessionID, correction, false)
    })

    const contextFor = async (sessionID: string) => {
      const [jobs, plan, current, all] = await Promise.all([
        ledger.forSession(sessionID),
        readPlan(sessionID),
        catalog(),
        options.maxUsdTotal === undefined ? undefined : ledger.all(),
      ])
      const totalSpentUsd = all && spentUsd(all)
      return sessionContext({ hfUser: user, limits: limitsFor(options, current), jobs, totalSpentUsd, plan })
    }

    await ctx.session.hook("context", async (event) => {
      if (event.agent !== ML_INTERN_AGENT) return
      event.system.push({ type: "text", text: await contextFor(event.sessionID) } as any)
    })

    await ctx.session.hook("compaction", async (event) => {
      if (event.agent !== ML_INTERN_AGENT) return
      event.system.push({ type: "text", text: COMPACTION_NOTE + "\n\n" + (await contextFor(event.sessionID)) } as any)
    })

    const watcher = new EventWatcher(
      options.dataDir,
      ledger,
      (record, jobEvent) => {
        polls.reset(record.sessionID, record.jobId)
        return notify(record.sessionID, formatEvent(jobEvent), true)
      },
      options.eventPollMs,
      warn,
    )
    await watcher.start()
    return () => watcher.stop()
  },
})
