import { describe, expect, it } from "vitest"
import plugin from "../src/index.js"

describe("plugin", () => {
  it("exposes the ml-intern id", () => {
    expect(plugin.id).toBe("ml-intern")
  })
})
