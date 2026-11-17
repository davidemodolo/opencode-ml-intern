import { appendFileSync, mkdirSync } from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "out")
mkdirSync(OUT, { recursive: true })
const LOG = join(OUT, process.env.SPIKE_LOG ?? "spike.log")
const log = (entry) => appendFileSync(LOG, JSON.stringify({ t: Date.now(), ...entry }) + "\n")
const errText = (e) => String(e?.message ?? e).slice(0, 300)

export default {
  id: "spike",
  async setup(ctx) {
    const s4 = process.env.SPIKE_S4 ?? "off"
    const echoCalls = new Map()
    log({ ev: "setup", s4 })

    await ctx.tool.transform((editor) => {
      editor.add({
        name: "spike_start_job",
        description: "Start a background job. Completion is notified to the session automatically; end your turn after calling it.",
        input: { type: "object", properties: { seconds: { type: "number" } }, required: ["seconds"], additionalProperties: false },
        execute: async (input, call) => {
          setTimeout(async () => {
            try {
              const r = await ctx.session.synthetic({
                sessionID: call.sessionID,
                text: "[job 7] finished with exit code 0. Reply with exactly the word JOBDONE.",
                resume: true,
              })
              log({ ev: "s1 synthetic ok", type: r?.type, delivery: r?.delivery })
            } catch (e) {
              log({ ev: "s1 synthetic err", m: errText(e) })
            }
          }, input.seconds * 1000)
          return { content: "job 7 started; you will be notified when it finishes" }
        },
      })
      editor.add({
        name: "spike_spend",
        description: "Spend compute credits. Requires approval.",
        input: { type: "object", properties: {}, additionalProperties: false },
        execute: async (_input, call) => {
          await ctx.permission.assert({
            sessionID: call.sessionID,
            source: { type: "tool", messageID: call.messageID, id: call.id },
            action: "spike_spend",
            resources: ["local:gpu0:2h"],
            save: ["local:*"],
            metadata: { description: "Estimated cost: 2 GPU hours on local gpu0" },
          })
          return { content: "spent" }
        },
      })
    })

    await ctx.skill.transform((editor) => {
      editor.add({
        id: "spike-skill",
        name: "spike-skill",
        description: "Spike skill with LoRA recipes",
        path: join(OUT, "SKILL.md"),
        content: "# spike-skill\nUse LoRA rank 16.",
      })
    })

    await ctx.tool.hook("execute.before", (e) => {
      if (e.tool !== "probe_echo") return
      const n = (echoCalls.get(e.sessionID) ?? 0) + 1
      echoCalls.set(e.sessionID, n)
      log({ ev: "s4 before", n })
      if (s4 === "before" && n >= 2) throw new Error("loop guard: probe_echo was already called with this input")
    })

    await ctx.permission.hook("evaluate", (e) => {
      log({ ev: "evaluate", agent: e.agent, action: e.action, resources: e.resources, effect: e.effect, metadata: e.metadata })
      if (e.action === "spike_spend") e.message = "Budget check: 2 of 10 GPU hours used"
      if (s4 === "evaluate" && e.action === "probe_echo") {
        const n = (echoCalls.get("eval:" + e.sessionID) ?? 0) + 1
        echoCalls.set("eval:" + e.sessionID, n)
        if (n >= 2) {
          e.effect = "deny"
          e.message = "loop guard: probe_echo was already called with this input"
        }
      }
    })

    await ctx.tool.hook("execute.after", (e) => {
      log({ ev: "after", agent: e.agent, tool: e.tool, status: e.status, error: e.error ? errText(e.error) : undefined })
    })

    await ctx.session.hook("context", (e) => {
      const text = JSON.stringify({ system: e.system, tools: e.tools })
      log({ ev: "context", agent: e.agent, tools: Object.keys(e.tools), skillVisible: text.includes("spike-skill"), jobDoneInMessages: JSON.stringify(e.messages).includes("JOBDONE") })
    })

    await ctx.session.hook("experimental.idle", (e) => {
      log({ ev: "idle", background: e.background.map((b) => b.kind) })
    })
  },
}
