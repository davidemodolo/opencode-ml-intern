import { describe, expect, it } from "vitest"
import { PollGuard } from "../src/poll-guard.js"

const running = (seconds: number) => ({ content: [{ type: "text", text: `{"job_id": "j1", "state": "running", "duration_seconds": ${seconds}}` }] })

describe("PollGuard", () => {
  it("blocks status polling of a running job after the threshold", () => {
    const guard = new PollGuard(3)
    for (let i = 0; i < 3; i++) {
      expect(guard.check("s", { job_id: "j1" })).toBeUndefined()
      guard.record("s", { job_id: "j1" }, running(i))
    }
    expect(guard.check("s", { job_id: "j1" })).toMatch(/Stop polling/)
    expect(guard.check("s", { job_id: "j2" })).toBeUndefined()
    expect(guard.check("other", { job_id: "j1" })).toBeUndefined()
  })

  it("resets when the job finishes or is notified", () => {
    const guard = new PollGuard(2)
    guard.record("s", { job_id: "j1" }, running(1))
    guard.record("s", { job_id: "j1" }, running(2))
    guard.record("s", { job_id: "j1" }, '{"job_id":"j1","state":"succeeded"}')
    expect(guard.check("s", { job_id: "j1" })).toBeUndefined()
    guard.record("s", { job_id: "j1" }, running(3))
    guard.record("s", { job_id: "j1" }, running(4))
    guard.reset("s", "j1")
    expect(guard.check("s", { job_id: "j1" })).toBeUndefined()
  })
})
