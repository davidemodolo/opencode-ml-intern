import { gateReason, type GateLimits, type JobLedger, type JobRecord } from "./ledger.js"

export const DEFAULT_TIMEOUT_MINUTES = 60
const RESERVATION_TTL_MS = 15 * 60_000

export const timeoutOf = (input: unknown) =>
  Number((input as { timeout_minutes?: unknown })?.timeout_minutes ?? DEFAULT_TIMEOUT_MINUTES)

interface Reservation {
  readonly sessionID: string
  readonly timeoutMinutes: number
  readonly at: number
}

/** Checks submit_job calls against the session limits, counting the calls that have not finished yet. */
export class SubmitGate {
  private readonly reservations = new Map<string, Reservation>()

  constructor(
    private readonly ledger: JobLedger,
    private readonly limits: GateLimits,
    private readonly now: () => number = Date.now,
  ) {}

  /** Returns why the call must be refused, or reserves its slot until release. */
  async reserve(callID: string, sessionID: string, timeoutMinutes: number): Promise<string | undefined> {
    const records = await this.ledger.forSession(sessionID)
    const reason = gateReason([...records, ...this.pending(sessionID)], timeoutMinutes, this.limits)
    if (!reason) this.reservations.set(callID, { sessionID, timeoutMinutes, at: this.now() })
    return reason
  }

  release(callID: string): void {
    this.reservations.delete(callID)
  }

  private pending(sessionID: string): JobRecord[] {
    // A call that never reaches execute.after (a denied approval) must not hold its slot forever.
    const cutoff = this.now() - RESERVATION_TTL_MS
    const records: JobRecord[] = []
    for (const [callID, r] of this.reservations) {
      if (r.at < cutoff) this.reservations.delete(callID)
      else if (r.sessionID === sessionID) {
        records.push({ jobId: callID, sessionID, name: callID, timeoutMinutes: r.timeoutMinutes, submittedAt: r.at })
      }
    }
    return records
  }
}
