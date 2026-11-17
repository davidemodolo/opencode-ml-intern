import type { KeyValueStore } from "../src/ledger.js"

export class MemoryStore implements KeyValueStore {
  readonly data = new Map<string, unknown>()
  async get(key: string) {
    return this.data.get(key)
  }
  async set(key: string, value: unknown) {
    this.data.set(key, structuredClone(value))
  }
  async scan({ prefix, after, limit = 100 }: { prefix: string; after?: string; limit?: number }) {
    const keys = [...this.data.keys()].filter((k) => k.startsWith(prefix) && (!after || k > after)).sort()
    const page = keys.slice(0, limit)
    return {
      entries: page.map((key) => ({ key, value: this.data.get(key) })),
      next: keys.length > limit ? page.at(-1) : undefined,
    }
  }
}
