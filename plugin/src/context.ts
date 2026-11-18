import { gpuHours, isFinal, isLocal, spentUsd, type JobRecord } from "./ledger.js"
import type { Limits } from "./gate.js"
import { formatPlan, type PlanItem } from "./plan.js"

const usd = (value: number) => `$${value.toFixed(2)}`

const describe = (job: JobRecord) => {
  const where = isLocal(job) ? "" : ` on ${job.backend}${job.hardware ? ` (${job.hardware})` : ""}`
  return `- ${job.jobId} (${job.name})${where}: ${job.state ?? "submitted"}`
}

export function sessionContext(input: {
  readonly hfUser: string | undefined
  readonly limits: Limits
  readonly jobs: readonly JobRecord[]
  readonly totalSpentUsd?: number
  readonly plan: readonly PlanItem[]
}): string {
  const { limits, jobs } = input
  const active = jobs.filter((j) => !isFinal(j))
  const recent = jobs.filter(isFinal).slice(-5)
  const spent = spentUsd(jobs)
  const lines = [
    "# ML Intern session context",
    `HF user: ${input.hfUser ?? "unknown"}`,
    `Local GPU hours used or reserved: ${gpuHours(jobs).toFixed(2)} of ${limits.maxGpuHours}`,
  ]
  if (limits.maxUsd > 0 || spent > 0) {
    const total =
      limits.maxUsdTotal !== undefined && input.totalSpentUsd !== undefined
        ? `, ${usd(input.totalSpentUsd)} of ${usd(limits.maxUsdTotal)} across sessions`
        : ""
    lines.push(`Paid backends budget: ${usd(spent)} of ${usd(limits.maxUsd)} spent or reserved in this session${total}`)
  }
  lines.push(`Active jobs: ${active.length} of ${limits.maxActiveJobs}`)
  for (const job of active) lines.push(describe(job))
  if (recent.length) {
    lines.push("Recently finished jobs:")
    for (const job of recent) lines.push(describe(job))
  }
  if (input.plan.length) lines.push("", formatPlan(input.plan))
  return lines.join("\n")
}
