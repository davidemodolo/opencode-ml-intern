import { describe, expect, it } from "vitest"
import { LoopGuard } from "../src/loop-guard.js"

describe("LoopGuard", () => {
  it("flags three identical calls with identical results and blocks the next one", () => {
    const guard = new LoopGuard(3)
    expect(guard.record("s", "read", { path: "a" }, "same")).toBeUndefined()
    expect(guard.record("s", "read", { path: "a" }, "same")).toBeUndefined()
    expect(guard.record("s", "read", { path: "a" }, "same")).toMatch(/REPETITION GUARD.*'read'/)
    expect(guard.check("s", "read", { path: "a" })).toMatch(/Blocked/)
  })

  it("treats key order as the same arguments", () => {
    const guard = new LoopGuard(2)
    guard.record("s", "t", { a: 1, b: 2 }, "r")
    expect(guard.record("s", "t", { b: 2, a: 1 }, "r")).toBeDefined()
  })

  it("does not flag polling whose result changes", () => {
    const guard = new LoopGuard(3)
    for (let i = 0; i < 6; i++) expect(guard.record("s", "job_status", { id: "1" }, `step ${i}`)).toBeUndefined()
  })

  it("flags a repeating cycle of two calls", () => {
    const guard = new LoopGuard(3)
    guard.record("s", "a", {}, "x")
    guard.record("s", "b", {}, "y")
    guard.record("s", "a", {}, "x")
    expect(guard.record("s", "b", {}, "y")).toMatch(/repeating cycle.*a -> b/)
  })

  it("lifts the block when a different call arrives and keeps sessions apart", () => {
    const guard = new LoopGuard(2)
    guard.record("s", "t", {}, "r")
    guard.record("s", "t", {}, "r")
    expect(guard.check("other", "t", {})).toBeUndefined()
    expect(guard.check("s", "u", {})).toBeUndefined()
    expect(guard.check("s", "t", {})).toBeUndefined()
  })
})
