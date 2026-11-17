import { mkdtemp, mkdir, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { EventWatcher, formatEvent, type JobEvent } from "../src/events.js"
import { JobLedger } from "../src/ledger.js"
import { MemoryStore } from "./memory-store.js"

const event: JobEvent = {
  job_id: "j1",
  name: "sft",
  backend: "local",
  state: "succeeded",
  exit_code: 0,
  duration_seconds: 42.4,
  error: null,
  log_tail: ["loss 0.5"],
}

describe("EventWatcher", () => {
  it("delivers each event once to the session that submitted the job", async () => {
    const dir = await mkdtemp(join(tmpdir(), "mli-"))
    await mkdir(join(dir, "events"))
    await writeFile(join(dir, "events", "j1.json"), JSON.stringify(event))
    await writeFile(join(dir, "events", "other.json"), JSON.stringify({ ...event, job_id: "other" }))
    const ledger = new JobLedger(new MemoryStore())
    await ledger.add({ jobId: "j1", sessionID: "s1", name: "sft", timeoutMinutes: 60, submittedAt: 0 })
    const delivered: string[] = []
    const watcher = new EventWatcher(dir, ledger, async (record, e) => void delivered.push(`${record.sessionID}:${e.job_id}`), 60_000)
    await watcher.start()
    await watcher.scan()
    watcher.stop()
    expect(delivered).toEqual(["s1:j1"])
    expect(await ledger.get("j1")).toMatchObject({ state: "succeeded", durationSeconds: 42.4, delivered: true })
  })

  it("delivers an event once when several watchers share the folder and the ledger", async () => {
    const dir = await mkdtemp(join(tmpdir(), "mli-"))
    await mkdir(join(dir, "events"))
    await writeFile(join(dir, "events", "j1.json"), JSON.stringify(event))
    const ledger = new JobLedger(new MemoryStore())
    await ledger.add({ jobId: "j1", sessionID: "s1", name: "sft", timeoutMinutes: 60, submittedAt: 0 })
    const delivered: string[] = []
    const watchers = [1, 2, 3].map(() => new EventWatcher(dir, ledger, async (r, e) => void delivered.push(e.job_id), 60_000))
    await Promise.all(watchers.map((w) => w.scan()))
    expect(delivered).toEqual(["j1"])
  })

  it("formats the notification with state, exit code and log tail", () => {
    expect(formatEvent(event)).toBe(
      "[ml-intern] Job j1 (sft) on local succeeded after 42 s, exit code 0.\nLast log lines:\n```\nloss 0.5\n```\nRead more with job_logs. Continue the task from here.",
    )
  })
})
