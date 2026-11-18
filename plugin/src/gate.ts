import type { Catalog } from "./catalog.js"
import { gpuHours, isFinal, isLocal, LOCAL_BACKEND, spentUsd, type JobLedger, type JobRecord } from "./ledger.js"

export const DEFAULT_TIMEOUT_MINUTES = 60
export const GPU_HOURS_PER_GPU = 24
const RESERVATION_TTL_MS = 15 * 60_000

export interface JobRequest {
  readonly backend: string
  readonly hardware?: string
  readonly gpus: number
  readonly timeoutMinutes: number
}

export function requestOf(input: unknown): JobRequest {
  const fields = (input ?? {}) as Record<string, unknown>
  const gpus = Number(fields.gpus ?? 1)
  return {
    backend: typeof fields.backend === "string" ? fields.backend : LOCAL_BACKEND,
    hardware: typeof fields.hardware === "string" ? fields.hardware : undefined,
    gpus: Number.isInteger(gpus) && gpus >= 1 ? gpus : 1,
    timeoutMinutes: Number(fields.timeout_minutes ?? DEFAULT_TIMEOUT_MINUTES),
  }
}

export interface GateOptions {
  readonly maxActiveJobs: number
  readonly maxGpuHours?: number
  readonly maxUsd: number
  readonly maxUsdTotal?: number
  readonly budgetAskRatio: number
}

export interface Limits extends GateOptions {
  readonly maxGpuHours: number
}

/** The GPU-hour limit defaults to GPU_HOURS_PER_GPU for each local GPU. */
export const limitsFor = (options: GateOptions, catalog: Catalog): Limits => ({
  ...options,
  maxGpuHours: options.maxGpuHours ?? GPU_HOURS_PER_GPU * Math.max(1, catalog.localGpus),
})

/** Hourly price times timeout for a paid backend, undefined when the catalog has no price for the hardware. */
export function estimateUsd(request: JobRequest, catalog: Catalog): number | undefined {
  const price = catalog.usdPerHour(request.backend, request.hardware)
  return price === undefined ? undefined : (price * request.timeoutMinutes) / 60
}

export type Decision = { readonly effect: "allow" } | { readonly effect: "ask" | "deny"; readonly reason: string }

const usd = (value: number) => `$${value.toFixed(2)}`

function budgetCheck(scope: string, spent: number, estimate: number, limit: number, ratio: number, option: string): Decision {
  const after = spent + estimate
  const summary = `${usd(spent)} of the ${scope} budget of ${usd(limit)} is spent or reserved and this job reserves ${usd(estimate)} (hourly price times timeout)`
  if (after > limit) {
    return { effect: "deny", reason: `Refused: ${summary}. Lower timeout_minutes, pick cheaper hardware or ask the user to raise ${option}.` }
  }
  if (after > ratio * limit) return { effect: "ask", reason: `Near the budget: ${summary}.` }
  return { effect: "allow" }
}

/** Checks a new job against the session limits and the dollar budgets. */
export function gateDecision(input: {
  readonly session: readonly JobRecord[]
  readonly all: readonly JobRecord[]
  readonly request: JobRequest
  readonly estimatedUsd: number | undefined
  readonly limits: Limits
}): Decision {
  const { session, all, request, estimatedUsd, limits } = input
  const active = session.filter((r) => !isFinal(r)).length
  if (active >= limits.maxActiveJobs) {
    return {
      effect: "deny",
      reason: `Refused: this session already has ${active} active jobs (limit ${limits.maxActiveJobs}). Wait for one to finish or cancel one.`,
    }
  }
  if (isLocal(request)) {
    const used = gpuHours(session)
    const requested = (request.gpus * request.timeoutMinutes) / 60
    if (used + requested <= limits.maxGpuHours) return { effect: "allow" }
    return {
      effect: "deny",
      reason:
        `Refused: the GPU-hour budget of this session is ${limits.maxGpuHours} h; ${used.toFixed(2)} h are used or reserved ` +
        `and this job reserves ${requested.toFixed(2)} h (GPUs times timeout). Lower timeout_minutes or ask the user to raise maxGpuHours.`,
    }
  }
  if (estimatedUsd === undefined) {
    return {
      effect: "deny",
      reason: `Refused: no price is known for hardware ${request.hardware ?? "(none)"} on backend ${request.backend}. Call list_hardware and set hardware to one of its options.`,
    }
  }
  const checks = [budgetCheck("session", spentUsd(session), estimatedUsd, limits.maxUsd, limits.budgetAskRatio, "maxUsd")]
  if (limits.maxUsdTotal !== undefined) {
    checks.push(budgetCheck("total", spentUsd(all), estimatedUsd, limits.maxUsdTotal, limits.budgetAskRatio, "maxUsdTotal"))
  }
  return checks.find((c) => c.effect === "deny") ?? checks.find((c) => c.effect === "ask") ?? { effect: "allow" }
}

/** Checks submit_job calls against the limits, counting the calls that have not finished yet. */
export class SubmitGate {
  private readonly reservations = new Map<string, JobRecord>()

  constructor(
    private readonly ledger: JobLedger,
    private readonly options: GateOptions,
    private readonly catalog: () => Promise<Catalog>,
    private readonly now: () => number = Date.now,
  ) {}

  /** Decides on the call and, unless it is denied, reserves its budget until release. */
  async reserve(callID: string, sessionID: string, request: JobRequest): Promise<Decision> {
    const catalog = await this.catalog()
    const estimatedUsd = estimateUsd(request, catalog)
    const needsAll = !isLocal(request) && this.options.maxUsdTotal !== undefined
    const [session, all] = await Promise.all([this.ledger.forSession(sessionID), needsAll ? this.ledger.all() : []])
    const pending = this.pending()
    const decision = gateDecision({
      session: [...session, ...pending.filter((r) => r.sessionID === sessionID)],
      all: [...all, ...pending],
      request,
      estimatedUsd,
      limits: limitsFor(this.options, catalog),
    })
    if (decision.effect !== "deny") {
      this.reservations.set(callID, {
        jobId: callID,
        sessionID,
        name: callID,
        submittedAt: this.now(),
        backend: request.backend,
        hardware: request.hardware,
        gpus: request.gpus,
        timeoutMinutes: request.timeoutMinutes,
        estimatedUsd,
      })
    }
    return decision
  }

  release(callID: string): void {
    this.reservations.delete(callID)
  }

  private pending(): JobRecord[] {
    // A call that never reaches execute.after (a denied approval) must not hold its slot forever.
    const cutoff = this.now() - RESERVATION_TTL_MS
    for (const [callID, r] of this.reservations) if (r.submittedAt < cutoff) this.reservations.delete(callID)
    return [...this.reservations.values()]
  }
}
