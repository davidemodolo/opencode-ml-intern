export interface KeyValueStore {
  get(key: string): Promise<unknown>
  set(key: string, value: any): Promise<void>
  scan(options: { prefix: string; after?: string; limit?: number }): Promise<{
    readonly entries: readonly { readonly key: string; readonly value: unknown }[]
    readonly next?: string
  }>
}

export const LOCAL_BACKEND = "local"

export interface JobRecord {
  readonly jobId: string
  readonly sessionID: string
  readonly name: string
  readonly timeoutMinutes: number
  readonly submittedAt: number
  readonly backend?: string
  readonly hardware?: string
  readonly gpus?: number
  readonly estimatedUsd?: number
  readonly costUsd?: number | null
  readonly state?: string
  readonly durationSeconds?: number | null
  readonly delivered?: boolean
}

const JOBS = "job/"
const SESSION_OF = "job-session/"
const jobKey = (sessionID: string, jobId: string) => `${JOBS}${sessionID}/${jobId}`
const FINAL_STATES = new Set(["succeeded", "failed", "timed_out", "cancelled"])

export const isFinal = (record: JobRecord) => record.state !== undefined && FINAL_STATES.has(record.state)
export const isLocal = (record: Pick<JobRecord, "backend">) => (record.backend ?? LOCAL_BACKEND) === LOCAL_BACKEND

/** Jobs submitted from opencode sessions, persisted in plugin storage under their session. */
export class JobLedger {
  constructor(private readonly store: KeyValueStore) {}

  async add(record: JobRecord): Promise<void> {
    await this.store.set(SESSION_OF + record.jobId, record.sessionID)
    await this.store.set(jobKey(record.sessionID, record.jobId), record as any)
  }

  async get(jobId: string): Promise<JobRecord | undefined> {
    const sessionID = (await this.store.get(SESSION_OF + jobId)) as string | undefined
    if (sessionID === undefined) return undefined
    return (await this.store.get(jobKey(sessionID, jobId))) as JobRecord | undefined
  }

  async update(jobId: string, patch: Partial<JobRecord>): Promise<JobRecord | undefined> {
    const current = await this.get(jobId)
    if (!current) return undefined
    const next = { ...current, ...patch }
    await this.store.set(jobKey(current.sessionID, jobId), next as any)
    return next
  }

  forSession(sessionID: string): Promise<JobRecord[]> {
    return this.list(jobKey(sessionID, ""))
  }

  all(): Promise<JobRecord[]> {
    return this.list(JOBS)
  }

  private async list(prefix: string): Promise<JobRecord[]> {
    const records: JobRecord[] = []
    let after: string | undefined
    do {
      const page = await this.store.scan({ prefix, after, limit: 200 })
      records.push(...page.entries.map((e) => e.value as JobRecord))
      after = page.next
    } while (after)
    return records.sort((a, b) => a.submittedAt - b.submittedAt)
  }
}

/** GPU hours of the local backend: finished jobs count their duration, active jobs their timeout, times their GPUs. */
export function gpuHours(records: readonly JobRecord[]): number {
  return records.filter(isLocal).reduce((total, r) => {
    const gpus = r.gpus ?? 1
    if (isFinal(r) && typeof r.durationSeconds === "number") return total + (gpus * r.durationSeconds) / 3600
    if (isFinal(r)) return total
    return total + (gpus * r.timeoutMinutes) / 60
  }, 0)
}

/** Dollars on paid backends: finished jobs count their cost, active jobs their estimate (price times timeout). */
export function spentUsd(records: readonly JobRecord[]): number {
  return records.reduce((total, r) => {
    if (isLocal(r)) return total
    const cost = isFinal(r) ? (r.costUsd ?? r.estimatedUsd) : r.estimatedUsd
    return total + (cost ?? 0)
  }, 0)
}
