import { describe, expect, it } from "vitest"
import { gpuHours, JobLedger, spentUsd, type JobRecord } from "../src/ledger.js"
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
    expect(await ledger.all()).toHaveLength(250)
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

describe("gpuHours", () => {
  it("counts local jobs, finished by duration and active by timeout, times their GPUs", () => {
    expect(gpuHours([job({ state: "succeeded", durationSeconds: 1800 }), job({ timeoutMinutes: 90 })])).toBe(2)
    expect(gpuHours([job({ gpus: 4, timeoutMinutes: 30 }), job({ backend: "hf_jobs", timeoutMinutes: 600 })])).toBe(2)
  })
})

describe("spentUsd", () => {
  it("counts paid jobs only", () => {
    expect(spentUsd([job({ estimatedUsd: 3 }), job({ backend: "skypilot", estimatedUsd: 1.5 })])).toBe(1.5)
  })
})
