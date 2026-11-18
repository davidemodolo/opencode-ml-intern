import { mkdtemp, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { Catalog } from "../src/catalog.js"
import { estimateUsd, gateDecision, limitsFor, requestOf, SubmitGate, type GateOptions, type JobRequest } from "../src/gate.js"
import { JobLedger, type JobRecord } from "../src/ledger.js"
import { MemoryStore } from "./memory-store.js"

const options: GateOptions = { maxActiveJobs: 1, maxUsd: 10, budgetAskRatio: 0.8 }
const catalog = new Catalog({ local: { gpus: 2 }, hf_jobs: { usd_per_hour: { "a10g-small": 1, "a100x8": 20 } } })
const local: JobRequest = { backend: "local", gpus: 1, timeoutMinutes: 60 }
const hf = (hardware: string, timeoutMinutes = 60): JobRequest => ({ backend: "hf_jobs", hardware, gpus: 1, timeoutMinutes })
const job = (patch: Partial<JobRecord>): JobRecord => ({ jobId: "j", sessionID: "s", name: "n", timeoutMinutes: 60, submittedAt: 0, ...patch })
const limits = limitsFor({ ...options, maxActiveJobs: 5 }, catalog)
const decide = (request: JobRequest, session: JobRecord[] = [], all: JobRecord[] = [], l = limits) =>
  gateDecision({ session, all, request, estimatedUsd: estimateUsd(request, catalog), limits: l })

describe("requestOf", () => {
  it("reads backend, hardware, GPUs and timeout from submit_job input", () => {
    expect(requestOf({ backend: "hf_jobs", hardware: "a10g-small", gpus: 2, timeout_minutes: 30 })).toEqual({
      backend: "hf_jobs",
      hardware: "a10g-small",
      gpus: 2,
      timeoutMinutes: 30,
    })
    expect(requestOf(undefined)).toEqual({ backend: "local", hardware: undefined, gpus: 1, timeoutMinutes: 60 })
    expect(requestOf({ gpus: "4" }).gpus).toBe(4)
    expect(requestOf({ gpus: "x" }).gpus).toBe(1)
  })
})

describe("limitsFor", () => {
  it("derives the GPU-hour limit from the local GPUs unless it is set", () => {
    expect(limitsFor(options, catalog).maxGpuHours).toBe(48)
    expect(limitsFor(options, new Catalog()).maxGpuHours).toBe(24)
    expect(limitsFor({ ...options, maxGpuHours: 5 }, catalog).maxGpuHours).toBe(5)
  })
})

describe("gateDecision", () => {
  it("counts local GPU hours as GPUs times timeout", () => {
    expect(decide({ ...local, gpus: 2, timeoutMinutes: 24 * 60 })).toEqual({ effect: "allow" })
    const denied = decide({ ...local, gpus: 2, timeoutMinutes: 24 * 60 }, [job({ gpus: 2, timeoutMinutes: 30 })])
    expect(denied).toMatchObject({ effect: "deny", reason: expect.stringMatching(/GPU-hour budget of this session is 48 h; 1.00 h/) })
  })

  it("leaves local GPU hours out of the dollar budget and paid jobs out of the GPU hours", () => {
    const paid = job({ backend: "hf_jobs", estimatedUsd: 9, timeoutMinutes: 48 * 60 })
    expect(decide(local, [paid])).toEqual({ effect: "allow" })
    expect(decide(hf("a10g-small"), [job({ timeoutMinutes: 47 * 60 })])).toEqual({ effect: "allow" })
  })

  it("allows below the ask ratio, asks near the budget and denies above it", () => {
    expect(decide(hf("a10g-small", 7 * 60))).toEqual({ effect: "allow" })
    expect(decide(hf("a10g-small", 9 * 60))).toMatchObject({ effect: "ask", reason: expect.stringMatching(/^Near the budget: \$0.00 of the session budget of \$10.00/) })
    expect(decide(hf("a100x8"))).toMatchObject({ effect: "deny", reason: expect.stringMatching(/reserves \$20.00.*maxUsd/) })
  })

  it("counts finished paid jobs by their cost and active ones by their estimate", () => {
    const records = [
      job({ backend: "hf_jobs", state: "succeeded", costUsd: 2, estimatedUsd: 5 }),
      job({ backend: "hf_jobs", state: "failed", costUsd: null, estimatedUsd: 1 }),
      job({ backend: "hf_jobs", estimatedUsd: 3 }),
    ]
    expect(decide(hf("a10g-small", 5 * 60), records)).toMatchObject({ effect: "deny", reason: expect.stringMatching(/\$6.00 of the session/) })
    expect(decide(hf("a10g-small", 150), records)).toMatchObject({ effect: "ask" })
    expect(decide(hf("a10g-small", 60), records)).toEqual({ effect: "allow" })
  })

  it("checks the budget across sessions when maxUsdTotal is set", () => {
    const total = { ...limits, maxUsdTotal: 20 }
    const others = [job({ sessionID: "other", backend: "hf_jobs", estimatedUsd: 19.5 })]
    expect(decide(hf("a10g-small"), [], others, total)).toMatchObject({ effect: "deny", reason: expect.stringMatching(/total budget.*maxUsdTotal/) })
    expect(decide(hf("a10g-small"), [], others)).toEqual({ effect: "allow" })
  })

  it("denies paid hardware without a known price and any job over the active limit", () => {
    expect(decide(hf("h100"))).toMatchObject({ effect: "deny", reason: expect.stringMatching(/no price is known for hardware h100 on backend hf_jobs.*list_hardware/) })
    expect(decide({ backend: "hf_jobs", gpus: 1, timeoutMinutes: 60 })).toMatchObject({ effect: "deny" })
    const full = limitsFor(options, catalog)
    expect(decide(local, [job({})], [], full)).toMatchObject({ effect: "deny", reason: expect.stringMatching(/1 active jobs/) })
  })
})

describe("SubmitGate", () => {
  const gateWith = (ledger = new JobLedger(new MemoryStore()), now: () => number = Date.now, opts = options) =>
    new SubmitGate(ledger, opts, async () => catalog, now)

  it("counts submissions that are still running against the limits", async () => {
    const gate = gateWith()
    const [first, second] = await Promise.all([gate.reserve("c1", "s1", local), gate.reserve("c2", "s1", local)])
    expect(first).toEqual({ effect: "allow" })
    expect(second).toMatchObject({ effect: "deny", reason: expect.stringMatching(/1 active jobs/) })
    expect((await gate.reserve("c3", "s2", local)).effect).toBe("allow")
    gate.release("c1")
    expect((await gate.reserve("c4", "s1", local)).effect).toBe("allow")
  })

  it("reserves the estimate of a paid job until release", async () => {
    const gate = gateWith(undefined, Date.now, { ...options, maxActiveJobs: 5 })
    expect((await gate.reserve("c1", "s1", hf("a10g-small", 6 * 60))).effect).toBe("allow")
    expect((await gate.reserve("c2", "s1", hf("a10g-small", 5 * 60))).effect).toBe("deny")
    gate.release("c1")
    expect((await gate.reserve("c3", "s1", hf("a10g-small", 5 * 60))).effect).toBe("allow")
  })

  it("adds reservations of other sessions to the total budget", async () => {
    const gate = gateWith(undefined, Date.now, { ...options, maxActiveJobs: 5, maxUsdTotal: 10 })
    expect((await gate.reserve("c1", "s1", hf("a10g-small", 6 * 60))).effect).toBe("allow")
    expect((await gate.reserve("c2", "s2", hf("a10g-small", 6 * 60))).effect).toBe("deny")
  })

  it("drops reservations whose call never finished", async () => {
    let now = 0
    const gate = gateWith(undefined, () => now)
    await gate.reserve("c1", "s1", local)
    now += 16 * 60_000
    expect((await gate.reserve("c2", "s1", local)).effect).toBe("allow")
  })
})

describe("Catalog", () => {
  it("reads GPU counts and prices from the server's catalog.json", async () => {
    const dir = await mkdtemp(join(tmpdir(), "mli-"))
    expect((await Catalog.read(dir)).localGpus).toBe(0)
    await writeFile(
      join(dir, "catalog.json"),
      JSON.stringify({ backends: { local: { gpus: 4, status: "ready", usd_per_hour: {} }, hf_jobs: { gpus: 0, usd_per_hour: { "t4-small": 0.4 } } } }),
    )
    const read = await Catalog.read(dir)
    expect(read.localGpus).toBe(4)
    expect(read.usdPerHour("hf_jobs", "t4-small")).toBe(0.4)
    expect(read.usdPerHour("hf_jobs", undefined)).toBeUndefined()
    expect(read.usdPerHour("skypilot", "A100:1")).toBeUndefined()
  })
})
