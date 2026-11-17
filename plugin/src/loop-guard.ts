import { createHash } from "node:crypto"

interface Signature {
  readonly name: string
  readonly args: string
  readonly result: string
}

const LOOKBACK = 30

function stable(value: unknown): string {
  if (Array.isArray(value)) return "[" + value.map(stable).join(",") + "]"
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    return "{" + entries.map(([k, v]) => JSON.stringify(k) + ":" + stable(v)).join(",") + "}"
  }
  return JSON.stringify(value) ?? "null"
}

export function hash(value: unknown): string {
  const text = typeof value === "string" ? value : stable(value)
  return createHash("md5").update(text).digest("hex").slice(0, 12)
}

const same = (a: Signature, b: Signature) => a.name === b.name && a.args === b.args && a.result === b.result

function identicalRun(signatures: readonly Signature[], threshold: number): Signature | undefined {
  let count = 1
  for (let i = 1; i < signatures.length; i++) {
    count = same(signatures[i]!, signatures[i - 1]!) ? count + 1 : 1
    if (count >= threshold) return signatures[i]
  }
  return undefined
}

function repeatingSequence(signatures: readonly Signature[]): Signature[] | undefined {
  const n = signatures.length
  for (let length = 2; length <= 5; length++) {
    if (n < length * 2) continue
    const pattern = signatures.slice(n - length)
    let reps = 0
    for (let start = n - length; start >= 0; start -= length) {
      const chunk = signatures.slice(start, start + length)
      if (chunk.every((s, i) => same(s, pattern[i]!))) reps++
      else break
    }
    if (reps >= 2) return pattern
  }
  return undefined
}

/** Detects repeated identical tool calls and repeating call cycles, as ml-intern's doom loop guard. */
export class LoopGuard {
  private readonly history = new Map<string, Signature[]>()
  private readonly blocked = new Map<string, Set<string>>()

  constructor(private readonly threshold = 3) {}

  /** Records a finished call. Returns a corrective message when a loop is detected. */
  record(sessionID: string, tool: string, input: unknown, result: unknown): string | undefined {
    const signatures = this.history.get(sessionID) ?? []
    signatures.push({ name: tool, args: hash(input ?? {}), result: hash(result ?? "") })
    if (signatures.length > LOOKBACK) signatures.splice(0, signatures.length - LOOKBACK)
    this.history.set(sessionID, signatures)

    const repeated = identicalRun(signatures, this.threshold)
    if (repeated) {
      this.block(sessionID, [repeated])
      return (
        `[REPETITION GUARD] You have called '${repeated.name}' with the same arguments ${this.threshold} times in a row, ` +
        "getting the same result each time. STOP repeating this approach: it is not working. " +
        "Step back and try a fundamentally different strategy: use a different tool, change your arguments " +
        "significantly, or explain to the user what you are stuck on and ask for guidance."
      )
    }
    const cycle = repeatingSequence(signatures)
    if (cycle) {
      this.block(sessionID, cycle)
      return (
        `[REPETITION GUARD] You are stuck in a repeating cycle of tool calls: [${cycle.map((s) => s.name).join(" -> ")}]. ` +
        "This pattern has repeated without progress. STOP this cycle and try a fundamentally different approach: " +
        "break the problem down differently, use other tools, or explain to the user what you are stuck on."
      )
    }
    return undefined
  }

  /** Returns a reason to block a call that repeats a detected loop; any other call lifts the block. */
  check(sessionID: string, tool: string, input: unknown): string | undefined {
    const blocked = this.blocked.get(sessionID)
    if (!blocked) return undefined
    if (blocked.has(tool + ":" + hash(input ?? {}))) {
      return `Blocked by the repetition guard: '${tool}' with these arguments keeps returning the same result. Change approach.`
    }
    this.blocked.delete(sessionID)
    return undefined
  }

  private block(sessionID: string, signatures: readonly Signature[]): void {
    this.blocked.set(sessionID, new Set(signatures.map((s) => s.name + ":" + s.args)))
    this.history.set(sessionID, [])
  }
}
