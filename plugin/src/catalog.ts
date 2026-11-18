import { readFile } from "node:fs/promises"
import { join } from "node:path"

interface BackendEntry {
  readonly gpus?: number
  readonly usd_per_hour?: Readonly<Record<string, number>>
}

/** GPU counts and hourly prices the MCP server saves in catalog.json each time it lists the hardware. */
export class Catalog {
  constructor(private readonly backends: Readonly<Record<string, BackendEntry>> = {}) {}

  static async read(dataDir: string): Promise<Catalog> {
    try {
      const data = JSON.parse(await readFile(join(dataDir, "catalog.json"), "utf8")) as { backends?: Record<string, BackendEntry> }
      return new Catalog(data.backends ?? {})
    } catch {
      return new Catalog()
    }
  }

  get localGpus(): number {
    return this.backends.local?.gpus ?? 0
  }

  usdPerHour(backend: string, hardware: string | undefined): number | undefined {
    if (hardware === undefined) return undefined
    const price = this.backends[backend]?.usd_per_hour?.[hardware]
    return typeof price === "number" ? price : undefined
  }
}
