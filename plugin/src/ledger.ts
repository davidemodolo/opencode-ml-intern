export interface KeyValueStore {
  get(key: string): Promise<unknown>
  set(key: string, value: any): Promise<void>
  scan(options: { prefix: string; after?: string; limit?: number }): Promise<{
    readonly entries: readonly { readonly key: string; readonly value: unknown }[]
    readonly next?: string
  }>
}

export interface JobRecord {
  readonly jobId: string
  readonly sessionID: string
  readonly name: string
  readonly timeoutMinutes: number
  readonly submittedAt: number
  readonly state?: string
  readonly durationSeconds?: number | null
  readonly delivered?: boolean
}

const PREFIX = "job/"
const FINAL_STATES = new Set(["succeeded", "failed", "timed_out", "cancelled"])

export const isFinal = (record: JobRecord) => record.state !== undefined && FINAL_STATES.has(record.state)

/** Jobs submitted from opencode sessions, persisted in plugin storage. */
export class JobLedger {
  constructor(private readonly store: KeyValueStore) {}

  async add(record: JobRecord): Promise<void> {
    await this.store.set(PREFIX + record.jobId, record as any)
  }

  async get(jobId: string): Promise<JobRecord | undefined> {
    return (await this.store.get(PREFIX + jobId)) as JobRecord | undefined
  }

  async update(jobId: string, patch: Partial<JobRecord>): Promise<JobRecord | undefined> {
    const current = await this.get(jobId)
    if (!current) return undefined
    const next = { ...current, ...patch }
    await this.store.set(PREFIX + jobId, next as any)
    return next
  }

  async all(): Promise<JobRecord[]> {
    const records: JobRecord[] = []
    let after: string | undefined
    do {
      const page = await this.store.scan({ prefix: PREFIX, after, limit: 200 })
      records.push(...page.entries.map((e) => e.value as JobRecord))
      after = page.next
    } while (after)
    return records
  }

  async forSession(sessionID: string): Promise<JobRecord[]> {
    return (await this.all()).filter((r) => r.sessionID === sessionID).sort((a, b) => a.submittedAt - b.submittedAt)
  }
}

export interface GateLimits {
  readonly maxActiveJobs: number
  readonly maxGpuHours: number
}

export function gpuHours(records: readonly JobRecord[]): number {
  return records.reduce((total, r) => {
    if (isFinal(r) && typeof r.durationSeconds === "number") return total + r.durationSeconds / 3600
    if (isFinal(r)) return total
    return total + r.timeoutMinutes / 60
  }, 0)
}

/** Returns why a new job must be refused, or undefined when it fits the session limits. */
export function gateReason(records: readonly JobRecord[], timeoutMinutes: number, limits: GateLimits): string | undefined {
  const active = records.filter((r) => !isFinal(r)).length
  if (active >= limits.maxActiveJobs) {
    return `Refused: this session already has ${active} active jobs (limit ${limits.maxActiveJobs}). Wait for one to finish or cancel one.`
  }
  const used = gpuHours(records)
  const requested = timeoutMinutes / 60
  if (used + requested > limits.maxGpuHours) {
    return (
      `Refused: the GPU-hour budget of this session is ${limits.maxGpuHours} h; ${used.toFixed(2)} h are used or reserved ` +
      `and this job reserves ${requested.toFixed(2)} h (its timeout). Lower timeout_minutes or ask the user to raise maxGpuHours.`
    )
  }
  return undefined
}
