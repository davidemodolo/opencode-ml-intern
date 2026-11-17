import type { Options } from "./options.js"
import { AUTONOMOUS_PROMPT, ML_INTERN_PROMPT, ML_RESEARCH_PROMPT } from "./prompts.js"

export const ML_INTERN_AGENT = "ml-intern"
export const ML_RESEARCH_AGENT = "ml-research"
export const AGENT_COLOR = "#2EC4B6"

export const RESEARCH_TOOLS = [
  "hf_papers",
  "explore_hf_docs",
  "fetch_hf_docs",
  "find_hf_api",
  "hf_inspect_dataset",
  "hf_repo_files",
  "github_find_examples",
  "github_list_repos",
  "github_read_file",
] as const

type Effect = "allow" | "ask" | "deny"
export interface PermissionRule {
  action: string
  resource: string
  effect: Effect
}

interface MutableAgent {
  name?: string
  color?: string
  mode?: string
  description?: string
  system?: string
  permissions?: PermissionRule[]
}

export interface AgentEditorLike {
  update(id: string, update: (agent: any) => void): void
}

const rule = (action: string, effect: Effect, resource = "*"): PermissionRule => ({ action, resource, effect })

export function researchPermissions(options: Options): PermissionRule[] {
  return [
    rule("*", "deny"),
    ...["read", "glob", "grep", "skill", "webfetch", "websearch", "execute"].map((a) => rule(a, "allow")),
    ...RESEARCH_TOOLS.map((tool) => rule(`${options.mcpServer}_${tool}`, "allow")),
  ]
}

export function internPermissions(options: Options): PermissionRule[] {
  return [
    rule(`${options.mcpServer}_submit_job`, options.autoApprove ? "allow" : "ask"),
    rule("subagent", "allow", ML_RESEARCH_AGENT),
  ]
}

export function configureAgents(editor: AgentEditorLike, options: Options): void {
  editor.update(ML_INTERN_AGENT, (agent: MutableAgent) => {
    agent.name = "ML Intern"
    agent.color = AGENT_COLOR
    agent.mode = "primary"
    agent.description = "ML engineering agent: researches recipes, writes training scripts and runs GPU jobs"
    agent.system = options.autonomous ? ML_INTERN_PROMPT + "\n" + AUTONOMOUS_PROMPT : ML_INTERN_PROMPT
    agent.permissions = [...(agent.permissions ?? []), ...internPermissions(options)]
  })
  editor.update(ML_RESEARCH_AGENT, (agent: MutableAgent) => {
    agent.name = "ML Research"
    agent.color = AGENT_COLOR
    agent.mode = "subagent"
    agent.description =
      "Read-only ML research: crawls papers and citation graphs, extracts training recipes, verifies datasets and finds working code"
    agent.system = ML_RESEARCH_PROMPT
    agent.permissions = researchPermissions(options)
  })
}
