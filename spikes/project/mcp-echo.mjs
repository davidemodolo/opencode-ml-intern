import { createInterface } from "node:readline"

const send = (message) => process.stdout.write(JSON.stringify(message) + "\n")
createInterface({ input: process.stdin }).on("line", (line) => {
  let m
  try { m = JSON.parse(line) } catch { return }
  if (m.method === "initialize") return send({ jsonrpc: "2.0", id: m.id, result: { protocolVersion: m.params?.protocolVersion ?? "2025-06-18", capabilities: { tools: {} }, serverInfo: { name: "probe", version: "0.0.0" } } })
  if (m.method === "tools/list") return send({ jsonrpc: "2.0", id: m.id, result: { tools: [{ name: "echo", description: "Echo text back", inputSchema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] } }] } })
  if (m.method === "tools/call") return send({ jsonrpc: "2.0", id: m.id, result: { content: [{ type: "text", text: "echo: " + (m.params?.arguments?.text ?? "") }] } })
  if (m.id !== undefined) send({ jsonrpc: "2.0", id: m.id, result: {} })
})
