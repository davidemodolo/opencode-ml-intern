import { readFile } from "node:fs/promises"
import { homedir } from "node:os"
import { join } from "node:path"

type Env = Readonly<Record<string, string | undefined>>

export async function hfToken(env: Env, home = homedir()): Promise<string | undefined> {
  if (env.HF_TOKEN) return env.HF_TOKEN
  const hfHome = env.HF_HOME || join(env.XDG_CACHE_HOME || join(home, ".cache"), "huggingface")
  try {
    return (await readFile(join(hfHome, "token"), "utf8")).trim() || undefined
  } catch {
    return undefined
  }
}

export async function hfUser(env: Env, fetchImpl: typeof fetch = fetch): Promise<string | undefined> {
  const token = await hfToken(env)
  if (!token) return undefined
  try {
    const response = await fetchImpl("https://huggingface.co/api/whoami-v2", {
      headers: { authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(10_000),
    })
    if (!response.ok) return undefined
    const body = (await response.json()) as { name?: string }
    return body.name
  } catch {
    return undefined
  }
}
