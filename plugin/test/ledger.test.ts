import { describe, expect, it } from "vitest"
import { gateReason, gpuHours, JobLedger, type JobRecord } from "../src/ledger.js"
import { MemoryStore } from "./memory-store.js"

const job = (patch: Partial<JobRecord>): JobRecord => ({
  jobId: "j",
  sessionID: "s",
  name: "n",
  timeoutMinutes: 60,
  submittedAt: 0,
  ...patch,
})

describe("JobLedger", () => {
  it("stores, updates and lists jobs per session across scan pages", async () => {
    const ledger = new JobLedger(new MemoryStore())
    for (let i = 0; i < 250; i++) await ledger.add(job({ jobId: `j${i}`, sessionID: i % 2 ? "a" : "b", submittedAt: i }))
    await ledger.update("j1", { state: "succeeded" })
    expect((await ledger.get("j1"))?.state).toBe("succeeded")
    expect(await ledger.forSession("a")).toHaveLength(125)
  })

  it("reads only the records of the requested session", async () => {
    const store = new MemoryStore()
    const ledger = new JobLedger(store)
    for (let i = 0; i < 10; i++) await ledger.add(job({ jobId: `j${i}`, sessionID: i ? "other" : "mine" }))
    const scanned: string[] = []
    const scan = store.scan.bind(store)
    store.scan = async (options) => {
      const page = await scan(options)
      scanned.push(...page.entries.map((e) => e.key))
      return page
    }
    expect((await ledger.forSession("mine")).map((r) => r.jobId)).toEqual(["j0"])
    expect(scanned).toHaveLength(1)
  })
})

describe("gateReason", () => {
  const limits = { maxActiveJobs: 2, maxGpuHours: 3 }

  it("counts finished jobs by duration and active jobs by timeout", () => {
    expect(gpuHours([job({ state: "succeeded", durationSeconds: 1800 }), job({ timeoutMinutes: 90 })])).toBe(2)
  })

  it("refuses when too many jobs are active", () => {
    expect(gateReason([job({}), job({})], 10, limits)).toMatch(/2 active jobs/)
  })

  it("refuses when the timeout would exceed the GPU-hour budget", () => {
    expect(gateReason([job({ state: "failed", durationSeconds: 7200 })], 90, limits)).toMatch(/budget/)
    expect(gateReason([job({ state: "failed", durationSeconds: 7200 })], 30, limits)).toBeUndefined()
  })
})
