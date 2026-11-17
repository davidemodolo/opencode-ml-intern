import { homedir } from "node:os"
import { join } from "node:path"

export interface Options {
  readonly mcpServer: string
  readonly dataDir: string
  readonly maxActiveJobs: number
  readonly maxGpuHours: number
  readonly autoApprove: boolean
  readonly autonomous: boolean
  readonly loopThreshold: number
  readonly eventPollMs: number
  readonly serverCommand: readonly string[]
}

export const SERVER_PACKAGE = "ml-intern-mcp==0.1.0"
export const DEFAULT_SERVER_COMMAND: readonly string[] = ["uvx", SERVER_PACKAGE]

type Env = Readonly<Record<string, string | undefined>>

export function defaultDataDir(env: Env, home = homedir()): string {
  if (env.ML_INTERN_DATA_DIR) return env.ML_INTERN_DATA_DIR
  return join(env.XDG_DATA_HOME || join(home, ".local", "share"), "ml-intern-mcp")
}

function number(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback
}

export function resolveOptions(raw: Readonly<Record<string, unknown>>, env: Env, home = homedir()): Options {
  return {
    mcpServer: typeof raw.mcpServer === "string" ? raw.mcpServer : "ml-intern",
    dataDir: typeof raw.dataDir === "string" ? raw.dataDir : defaultDataDir(env, home),
    maxActiveJobs: number(raw.maxActiveJobs, 4),
    maxGpuHours: number(raw.maxGpuHours, 24),
    autoApprove: raw.autoApprove === true,
    autonomous: raw.autonomous === true,
    loopThreshold: number(raw.loopThreshold, 3),
    eventPollMs: number(raw.eventPollMs, 3000),
    serverCommand:
      Array.isArray(raw.serverCommand) && raw.serverCommand.every((part) => typeof part === "string")
        ? raw.serverCommand
        : DEFAULT_SERVER_COMMAND,
  }
}
