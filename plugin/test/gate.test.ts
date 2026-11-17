import { describe, expect, it } from "vitest"
import { SubmitGate } from "../src/gate.js"
import { JobLedger } from "../src/ledger.js"
import { MemoryStore } from "./memory-store.js"

const limits = { maxActiveJobs: 1, maxGpuHours: 24 }

describe("SubmitGate", () => {
  it("counts submissions that are still running against the limits", async () => {
    const gate = new SubmitGate(new JobLedger(new MemoryStore()), limits)
    const [first, second] = await Promise.all([gate.reserve("c1", "s1", 60), gate.reserve("c2", "s1", 60)])
    expect(first).toBeUndefined()
    expect(second).toMatch(/1 active jobs/)
    expect(await gate.reserve("c3", "s2", 60)).toBeUndefined()
    gate.release("c1")
    expect(await gate.reserve("c4", "s1", 60)).toBeUndefined()
  })

  it("drops reservations whose call never finished", async () => {
    let now = 0
    const gate = new SubmitGate(new JobLedger(new MemoryStore()), limits, () => now)
    await gate.reserve("c1", "s1", 60)
    now += 16 * 60_000
    expect(await gate.reserve("c2", "s1", 60)).toBeUndefined()
  })
})
