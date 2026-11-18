import { describe, expect, it } from "vitest"
import { sessionContext } from "../src/context.js"
import { formatPlan, validatePlan } from "../src/plan.js"

describe("plan", () => {
  it("validates and formats a plan", () => {
    const items = validatePlan({
      todos: [
        { id: "1", content: "Research", status: "completed" },
        { id: "2", content: "Train", status: "in_progress" },
      ],
    })
    expect(formatPlan(items)).toBe("Plan (1/2 completed):\n[x] Research\n[~] Train")
  })

  it("rejects missing fields and unknown statuses", () => {
    expect(() => validatePlan({ todos: [{ id: "1", content: "x" }] })).toThrow(/status/)
    expect(() => validatePlan({ todos: [{ id: "1", content: "x", status: "done" }] })).toThrow(/Invalid status/)
    expect(() => validatePlan({})).toThrow(/array/)
  })
})

describe("sessionContext", () => {
  const limits = { maxActiveJobs: 2, maxGpuHours: 10, maxUsd: 25, budgetAskRatio: 0.8 }

  it("lists budgets, active jobs and plan", () => {
    const text = sessionContext({
      hfUser: "alice",
      limits,
      jobs: [
        { jobId: "j1", sessionID: "s", name: "sft", timeoutMinutes: 120, submittedAt: 0 },
        { jobId: "j2", sessionID: "s", name: "big", timeoutMinutes: 60, submittedAt: 1, backend: "hf_jobs", hardware: "a100x8", estimatedUsd: 20 },
      ],
      plan: [{ id: "1", content: "Train", status: "in_progress" }],
    })
    expect(text).toContain("HF user: alice")
    expect(text).toContain("Local GPU hours used or reserved: 2.00 of 10")
    expect(text).toContain("Paid backends budget: $20.00 of $25.00 spent or reserved in this session\n")
    expect(text).toContain("- j1 (sft): submitted")
    expect(text).toContain("- j2 (big) on hf_jobs (a100x8): submitted")
    expect(text).toContain("[~] Train")
  })

  it("leaves the paid budget out when there is none and nothing was spent", () => {
    const text = sessionContext({ hfUser: undefined, limits: { ...limits, maxUsd: 0 }, jobs: [], plan: [] })
    expect(text).not.toContain("Paid backends")
  })

  it("adds the spend across sessions when a total budget is set", () => {
    const text = sessionContext({ hfUser: undefined, limits: { ...limits, maxUsdTotal: 100 }, jobs: [], totalSpentUsd: 42, plan: [] })
    expect(text).toContain("$0.00 of $25.00 spent or reserved in this session, $42.00 of $100.00 across sessions")
  })
})
