import { gpuHours, isFinal, type GateLimits, type JobRecord } from "./ledger.js"
import { formatPlan, type PlanItem } from "./plan.js"

export function sessionContext(input: {
  readonly hfUser: string | undefined
  readonly limits: GateLimits
  readonly jobs: readonly JobRecord[]
  readonly plan: readonly PlanItem[]
}): string {
  const active = input.jobs.filter((j) => !isFinal(j))
  const recent = input.jobs.filter(isFinal).slice(-5)
  const lines = [
    "# ML Intern session context",
    `HF user: ${input.hfUser ?? "unknown"}`,
    `GPU hours used or reserved: ${gpuHours(input.jobs).toFixed(2)} of ${input.limits.maxGpuHours}`,
    `Active jobs: ${active.length} of ${input.limits.maxActiveJobs}`,
  ]
  for (const job of active) lines.push(`- ${job.jobId} (${job.name}): ${job.state ?? "submitted"}`)
  if (recent.length) {
    lines.push("Recently finished jobs:")
    for (const job of recent) lines.push(`- ${job.jobId} (${job.name}): ${job.state}`)
  }
  if (input.plan.length) lines.push("", formatPlan(input.plan))
  return lines.join("\n")
}
