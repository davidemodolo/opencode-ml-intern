#!/usr/bin/env node
// Acceptance run: starts an isolated opencode server, asks the ml-intern agent for a LoRA smoke
// fine-tune on the local GPU, waits for the job notifications and checks that an adapter was saved.
import { spawn, execFileSync } from "node:child_process"
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync } from "node:fs"
import { join, dirname } from "node:path"
import { fileURLToPath } from "node:url"

const here = dirname(fileURLToPath(import.meta.url))
const fresh = process.argv.includes("--fresh")
const port = process.env.E2E_PORT ?? "4799"
const model = process.env.E2E_MODEL ?? "opencode/deepseek-v4.1-flash-free"
const dataDir = join(here, ".data")
const configHome = join(here, ".config")
const url = `http://127.0.0.1:${port}`
const env = {
  ...process.env,
  PWD: here,
  XDG_CONFIG_HOME: configHome,
  ML_INTERN_DATA_DIR: dataDir,
  OPENCODE_PASSWORD: process.env.OPENCODE_PASSWORD ?? "ml-intern-e2e",
}
const prompt =
  "Fine-tune Qwen/Qwen2.5-0.5B-Instruct with a LoRA adapter on the 'messages' column of trl-lib/Capybara " +
  "on the local GPU. This is a smoke run: limit training to 10 steps and a 20 minute timeout, save the adapter " +
  "under ./outputs and do not push to the Hub. Use the lora-sft skill as the starting point."

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const log = (...args) => console.log("[e2e]", ...args)

function jobs() {
  const root = join(dataDir, "jobs")
  if (!existsSync(root)) return []
  return readdirSync(root).map((id) => JSON.parse(readFileSync(join(root, id, "job.json"), "utf8")))
}

function adapterSaved(job) {
  return existsSync(join(dataDir, "jobs", job.id, "outputs", "adapter", "adapter_config.json"))
}

async function waitForAgent() {
  for (let attempt = 0; attempt < 60; attempt++) {
    await sleep(2000)
    try {
      const out = execFileSync("opencode", ["api", "agent.list", "--server", url], { cwd: here, env, stdio: ["ignore", "pipe", "ignore"] })
      if (JSON.parse(out.toString()).data.some((agent) => agent.id === "ml-intern")) return
    } catch {}
  }
  throw new Error("the ml-intern agent did not appear on the server")
}

if (fresh) rmSync(dataDir, { recursive: true, force: true })
mkdirSync(join(configHome, "opencode"), { recursive: true })

const server = spawn("opencode", ["serve", "--port", port, "--hostname", "127.0.0.1"], { cwd: here, env, stdio: "ignore" })
let exitCode = 1
try {
  await waitForAgent()
  log("running the ml-intern agent with", model)
  execFileSync(
    "opencode",
    ["run", "--server", url, "--agent", "ml-intern", "--archived", "--title", "e2e-lora", "-m", model, prompt],
    { cwd: here, env, stdio: "inherit", timeout: 20 * 60_000 },
  )
  const deadline = Date.now() + 40 * 60_000
  while (Date.now() < deadline) {
    const all = jobs()
    const running = all.filter((j) => j.state === "queued" || j.state === "running")
    const done = all.find((j) => j.state === "succeeded" && adapterSaved(j))
    if (done && running.length === 0) {
      log(`job ${done.id} saved an adapter`)
      exitCode = 0
      break
    }
    if (all.length && running.length === 0 && !done) log("no running jobs yet, waiting for the agent to react")
    await sleep(15_000)
  }
  await sleep(30_000)
  if (exitCode !== 0) log("no job saved an adapter before the deadline")
} finally {
  server.kill("SIGTERM")
}
process.exit(exitCode)
