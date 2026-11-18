import { watch, type FSWatcher } from "node:fs"
import { mkdir, open, readdir, readFile, rm } from "node:fs/promises"
import { join } from "node:path"
import type { JobLedger, JobRecord } from "./ledger.js"

export interface JobEvent {
  readonly job_id: string
  readonly name: string
  readonly backend: string
  readonly hardware?: string | null
  readonly state: string
  readonly exit_code: number | null
  readonly duration_seconds: number | null
  readonly cost_usd?: number | null
  readonly error: string | null
  readonly workdir?: string
  readonly log_tail: readonly string[]
}

export function formatEvent(event: JobEvent): string {
  const duration = event.duration_seconds == null ? "" : ` after ${Math.round(event.duration_seconds)} s`
  const exit = event.exit_code == null ? "" : `, exit code ${event.exit_code}`
  const cost = event.cost_usd == null ? "" : `, cost $${event.cost_usd.toFixed(2)}`
  const where = event.hardware ? `${event.backend} (${event.hardware})` : event.backend
  const lines = [`[ml-intern] Job ${event.job_id} (${event.name}) on ${where} ${event.state}${duration}${exit}${cost}.`]
  if (event.error) lines.push(`Error: ${event.error}`)
  if (event.workdir) lines.push(`Job directory (outputs are relative to it): ${event.workdir}`)
  if (event.log_tail.length) lines.push("Last log lines:", "```", ...event.log_tail, "```")
  lines.push("Read more with job_logs. Continue the task from here.")
  return lines.join("\n")
}

export type Deliver = (record: JobRecord, event: JobEvent) => Promise<void>

/**
 * Watches the MCP server's events folder and delivers each completion once to the session that submitted the job,
 * then removes the delivered event.
 */
export class EventWatcher {
  private readonly eventsDir: string
  private readonly inFlight = new Set<string>()
  private watcher: FSWatcher | undefined
  private timer: ReturnType<typeof setInterval> | undefined

  constructor(
    dataDir: string,
    private readonly ledger: JobLedger,
    private readonly deliver: Deliver,
    private readonly pollMs: number,
    private readonly onError: (error: unknown) => void = () => {},
  ) {
    this.eventsDir = join(dataDir, "events")
  }

  async start(): Promise<void> {
    await mkdir(this.eventsDir, { recursive: true })
    try {
      this.watcher = watch(this.eventsDir, () => void this.scan())
    } catch (error) {
      this.onError(error)
    }
    this.timer = setInterval(() => void this.scan(), this.pollMs)
    this.timer.unref?.()
    await this.scan()
  }

  stop(): void {
    this.watcher?.close()
    if (this.timer) clearInterval(this.timer)
  }

  async scan(): Promise<void> {
    let files: string[]
    try {
      files = (await readdir(this.eventsDir)).filter((f) => f.endsWith(".json"))
    } catch (error) {
      this.onError(error)
      return
    }
    await Promise.all(files.map((file) => this.process(file.slice(0, -".json".length))))
  }

  /** Several plugin instances watch the same folder: the exclusive create lets one of them deliver. */
  private async claim(jobId: string): Promise<boolean> {
    try {
      await (await open(join(this.eventsDir, jobId + ".delivered"), "wx")).close()
      return true
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "EEXIST") return false
      throw error
    }
  }

  private async process(jobId: string): Promise<void> {
    if (this.inFlight.has(jobId)) return
    this.inFlight.add(jobId)
    try {
      const record = await this.ledger.get(jobId)
      if (!record || record.delivered || !(await this.claim(jobId))) return
      const event = JSON.parse(await readFile(join(this.eventsDir, jobId + ".json"), "utf8")) as JobEvent
      const updated = await this.ledger.update(jobId, {
        state: event.state,
        durationSeconds: event.duration_seconds,
        costUsd: event.cost_usd ?? null,
        delivered: true,
      })
      await this.deliver(updated ?? record, event)
      await rm(join(this.eventsDir, jobId + ".json"), { force: true })
    } catch (error) {
      this.onError(error)
    } finally {
      this.inFlight.delete(jobId)
    }
  }
}
