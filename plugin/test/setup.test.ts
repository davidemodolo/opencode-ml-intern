import { describe, expect, it } from "vitest"
import { configureAgents, ML_INTERN_AGENT, ML_RESEARCH_AGENT } from "../src/agents.js"
import { jobIdFrom } from "../src/index.js"
import { registerServer } from "../src/mcp.js"
import { DEFAULT_SERVER_COMMAND, resolveOptions } from "../src/options.js"
import { loadSkills } from "../src/skills.js"

describe("options", () => {
  it("uses defaults and the same data dir as the server", () => {
    const options = resolveOptions({}, { XDG_DATA_HOME: "/data" }, "/home/u")
    expect(options).toMatchObject({ mcpServer: "ml-intern", dataDir: "/data/ml-intern-mcp", maxActiveJobs: 4 })
    expect(resolveOptions({}, {}, "/home/u").dataDir).toBe("/home/u/.local/share/ml-intern-mcp")
    expect(resolveOptions({ maxGpuHours: 2, autoApprove: true }, {}).maxGpuHours).toBe(2)
  })
})

describe("agents", () => {
  it("creates ml-intern with the submit gate and a read-only ml-research", () => {
    const agents: Record<string, any> = {}
    configureAgents(
      { update: (id, fn) => fn((agents[id] ??= { permissions: [{ action: "*", resource: "*", effect: "allow" }] })) },
      resolveOptions({}, {}),
    )
    expect(agents[ML_INTERN_AGENT].mode).toBe("primary")
    expect(agents[ML_INTERN_AGENT].permissions).toContainEqual({ action: "ml-intern_submit_job", resource: "*", effect: "ask" })
    const research = agents[ML_RESEARCH_AGENT]
    expect(research.mode).toBe("subagent")
    expect(research.permissions[0]).toEqual({ action: "*", resource: "*", effect: "deny" })
    expect(research.permissions).toContainEqual({ action: "ml-intern_hf_papers", resource: "*", effect: "allow" })
    expect(research.permissions.some((p: any) => p.action.endsWith("submit_job"))).toBe(false)
  })
})

describe("mcp server registration", () => {
  it("adds the server with the shared data dir unless one is configured", () => {
    const servers = new Map<string, any>()
    const editor = { get: (name: string) => servers.get(name), set: (name: string, config: any) => void servers.set(name, config) }
    const options = resolveOptions({}, { ML_INTERN_DATA_DIR: "/d" })
    expect(registerServer(editor, options)).toBe(true)
    expect(servers.get("ml-intern")).toEqual({
      type: "local",
      command: [...DEFAULT_SERVER_COMMAND],
      environment: { ML_INTERN_DATA_DIR: "/d" },
    })
    expect(registerServer(editor, options)).toBe(false)
  })
})

describe("skills", () => {
  it("loads every bundled skill with a description", async () => {
    const skills = await loadSkills()
    expect(skills.map((s) => s.id).sort()).toEqual(["lora-sft", "ml-job-script", "preference-tuning", "qlora"])
    for (const skill of skills) expect(skill.description.length).toBeGreaterThan(20)
  })
})

describe("jobIdFrom", () => {
  it("finds the job id in plain and nested JSON results", () => {
    expect(jobIdFrom({ content: [{ type: "text", text: '{"job_id": "20261117-1-abc"}' }] })).toBe("20261117-1-abc")
    expect(jobIdFrom('{"job_id":"x-1"}')).toBe("x-1")
    expect(jobIdFrom({})).toBeUndefined()
  })
})
