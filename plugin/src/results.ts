const field = (name: string) => new RegExp(name + '\\\\*"\\s*:\\s*\\\\*"([A-Za-z0-9_-]+)')

function text(result: unknown): string {
  return (typeof result === "string" ? result : JSON.stringify(result)) ?? ""
}

/** Reads a string field from an MCP tool result, whether plain or nested JSON text. */
export function resultField(result: unknown, name: string): string | undefined {
  return field(name).exec(text(result))?.[1]
}

export const jobIdFrom = (result: unknown) => resultField(result, "job_id")
