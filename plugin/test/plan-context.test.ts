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
  it("lists budget, active jobs and plan", () => {
    const text = sessionContext({
      hfUser: "alice",
      limits: { maxActiveJobs: 2, maxGpuHours: 10 },
      jobs: [{ jobId: "j1", sessionID: "s", name: "sft", timeoutMinutes: 120, submittedAt: 0 }],
      plan: [{ id: "1", content: "Train", status: "in_progress" }],
    })
    expect(text).toContain("HF user: alice")
    expect(text).toContain("GPU hours used or reserved: 2.00 of 10")
    expect(text).toContain("- j1 (sft): submitted")
    expect(text).toContain("[~] Train")
  })
})
