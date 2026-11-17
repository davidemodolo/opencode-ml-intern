import { readdir, readFile } from "node:fs/promises"
import { join } from "node:path"
import { fileURLToPath } from "node:url"

export interface SkillFile {
  readonly id: string
  readonly name: string
  readonly description: string
  readonly path: string
  readonly content: string
}

export const SKILLS_DIR = fileURLToPath(new URL("../skills/", import.meta.url))

export function parseSkill(id: string, path: string, text: string): SkillFile {
  const match = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/.exec(text)
  if (!match) throw new Error(`skill ${id} has no frontmatter`)
  const fields = Object.fromEntries(
    match[1]!.split("\n").map((line) => {
      const index = line.indexOf(":")
      return [line.slice(0, index).trim(), line.slice(index + 1).trim()]
    }),
  )
  if (!fields.description) throw new Error(`skill ${id} has no description`)
  return { id, name: fields.name || id, description: fields.description, path, content: match[2]!.trim() }
}

export async function loadSkills(dir = SKILLS_DIR): Promise<SkillFile[]> {
  const entries = await readdir(dir, { withFileTypes: true })
  return Promise.all(
    entries
      .filter((entry) => entry.isDirectory())
      .map(async (entry) => {
        const path = join(dir, entry.name, "SKILL.md")
        return parseSkill(entry.name, path, await readFile(path, "utf8"))
      }),
  )
}
