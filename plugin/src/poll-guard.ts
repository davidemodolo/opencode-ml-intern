import { resultField } from "./results.js"

const ACTIVE_STATES = new Set(["queued", "running"])

/** Stops repeated status polling of a job that is still running: its completion is notified to the session. */
export class PollGuard {
  private readonly polls = new Map<string, number>()

  constructor(private readonly threshold = 3) {}

  private key(sessionID: string, jobId: string) {
    return sessionID + ":" + jobId
  }

  record(sessionID: string, input: unknown, result: unknown): void {
    const jobId = (input as { job_id?: string })?.job_id
    if (!jobId) return
    const state = resultField(result, "state")
    const key = this.key(sessionID, jobId)
    if (state && ACTIVE_STATES.has(state)) this.polls.set(key, (this.polls.get(key) ?? 0) + 1)
    else this.polls.delete(key)
  }

  check(sessionID: string, input: unknown): string | undefined {
    const jobId = (input as { job_id?: string })?.job_id
    if (!jobId || (this.polls.get(this.key(sessionID, jobId)) ?? 0) < this.threshold) return undefined
    return (
      `Job ${jobId} is still running and its completion will be notified to this session automatically. ` +
      "Stop polling: continue with other work or end your turn."
    )
  }

  reset(sessionID: string, jobId: string): void {
    this.polls.delete(this.key(sessionID, jobId))
  }
}
