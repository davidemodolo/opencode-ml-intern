import { Plugin } from "@opencode/plugin"
import { configureAgents, ML_INTERN_AGENT, ML_RESEARCH_AGENT } from "./agents.js"
import { sessionContext } from "./context.js"
import { EventWatcher, formatEvent } from "./events.js"
import { hfUser } from "./hf.js"
import { gateReason, JobLedger } from "./ledger.js"
import { LoopGuard } from "./loop-guard.js"
import { resolveOptions } from "./options.js"
import { formatPlan, PLAN_TOOL_DESCRIPTION, PLAN_TOOL_INPUT, validatePlan, type PlanItem } from "./plan.js"
import { loadSkills } from "./skills.js"

const GUARDED_AGENTS = new Set([ML_INTERN_AGENT, ML_RESEARCH_AGENT])
const CODE_MODE_TOOL = "execute"
const COMPACTION_NOTE =
  "When summarizing this ML Intern session, keep the current plan, every job ID with its state, " +
  "the datasets and models in use and the results obtained so far."

export function jobIdFrom(result: unknown): string | undefined {
  const text = typeof result === "string" ? result : JSON.stringify(result)
  return /job_id\\*"\s*:\s*\\*"([A-Za-z0-9_-]+)/.exec(text ?? "")?.[1]
}

const warn = (error: unknown) => console.warn("[opencode-ml-intern]", error)

export default Plugin.define({
  id: "ml-intern",
  async setup(ctx) {
    const options = resolveOptions(ctx.options ?? {}, process.env)
    const submitTool = `${options.mcpServer}_submit_job`
    const ledger = new JobLedger(ctx.storage)
    const guard = new LoopGuard(options.loopThreshold)
    const skills = await loadSkills()
    let user: string | undefined
    void hfUser(process.env).then((name) => (user = name))

    const planKey = (sessionID: string) => `plan/${sessionID}`
    const readPlan = async (sessionID: string) => ((await ctx.storage.get(planKey(sessionID))) ?? []) as unknown as PlanItem[]
    const notify = (sessionID: string, text: string, resume: boolean) =>
      ctx.session.synthetic({ sessionID: sessionID as any, text, resume }).then(() => undefined, warn)

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
        const reason = guard.check(event.sessionID, event.tool, event.input)
        if (reason) throw new Error(reason)
      }
      if (event.tool === submitTool) {
        const timeout = Number((event.input as { timeout_minutes?: unknown })?.timeout_minutes ?? 60)
        const reason = gateReason(await ledger.forSession(event.sessionID), timeout, options)
        if (reason) throw new Error(reason)
      }
    })

    await ctx.tool.hook("execute.after", async (event) => {
      if (event.tool === CODE_MODE_TOOL) return
      if (event.tool === submitTool && event.status === "completed") {
        const jobId = jobIdFrom(event.result)
        if (jobId) {
          const input = (event.input ?? {}) as { name?: string; timeout_minutes?: number }
          await ledger.add({
            jobId,
            sessionID: event.sessionID,
            name: input.name ?? jobId,
            timeoutMinutes: input.timeout_minutes ?? 60,
            submittedAt: Date.now(),
          })
        }
      }
      if (!GUARDED_AGENTS.has(event.agent)) return
      const outcome = event.status === "completed" ? event.result : event.error
      const correction = guard.record(event.sessionID, event.tool, event.input, outcome)
      if (correction) await notify(event.sessionID, correction, false)
    })

    await ctx.session.hook("context", async (event) => {
      if (event.agent !== ML_INTERN_AGENT) return
      const [jobs, plan] = await Promise.all([ledger.forSession(event.sessionID), readPlan(event.sessionID)])
      event.system.push({ type: "text", text: sessionContext({ hfUser: user, limits: options, jobs, plan }) } as any)
    })

    await ctx.session.hook("compaction", async (event) => {
      if (event.agent !== ML_INTERN_AGENT) return
      const [jobs, plan] = await Promise.all([ledger.forSession(event.sessionID), readPlan(event.sessionID)])
      const context = sessionContext({ hfUser: user, limits: options, jobs, plan })
      event.system.push({ type: "text", text: COMPACTION_NOTE + "\n\n" + context } as any)
    })

    const watcher = new EventWatcher(
      options.dataDir,
      ledger,
      (record, jobEvent) => notify(record.sessionID, formatEvent(jobEvent), true),
      options.eventPollMs,
      warn,
    )
    await watcher.start()
    return () => watcher.stop()
  },
})
