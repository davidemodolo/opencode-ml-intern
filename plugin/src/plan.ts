export type PlanStatus = "pending" | "in_progress" | "completed"

export interface PlanItem {
  readonly id: string
  readonly content: string
  readonly status: PlanStatus
}

const STATUSES: readonly PlanStatus[] = ["pending", "in_progress", "completed"]
const MARK: Record<PlanStatus, string> = { pending: "[ ]", in_progress: "[~]", completed: "[x]" }

export const PLAN_TOOL_DESCRIPTION =
  "Track progress on multi-step tasks with a todo list (pending/in_progress/completed).\n\n" +
  "Use for tasks with 3+ steps. Each call replaces the entire plan (send the full list).\n\n" +
  "Rules: exactly ONE task in_progress at a time. Mark completed immediately after finishing. " +
  "Only mark completed when the task fully succeeded: keep in_progress if there are errors. " +
  "Update frequently so the user sees progress."

export const PLAN_TOOL_INPUT = {
  type: "object",
  properties: {
    todos: {
      type: "array",
      description: "List of todo items",
      items: {
        type: "object",
        properties: {
          id: { type: "string", description: "Unique identifier for the todo" },
          content: { type: "string", description: "Description of the todo task" },
          status: { type: "string", enum: STATUSES, description: "Current status of the todo" },
        },
        required: ["id", "content", "status"],
      },
    },
  },
  required: ["todos"],
} as const

export function validatePlan(input: unknown): PlanItem[] {
  const todos = (input as { todos?: unknown })?.todos
  if (!Array.isArray(todos)) throw new Error("todos must be an array. Call the tool again with the correct format.")
  return todos.map((todo, index) => {
    const item = todo as Partial<PlanItem>
    for (const field of ["id", "content", "status"] as const) {
      if (typeof item?.[field] !== "string") {
        throw new Error(`Todo ${index} is missing the required field '${field}'. Call the tool again with the correct format.`)
      }
    }
    if (!STATUSES.includes(item.status as PlanStatus)) {
      throw new Error(`Invalid status '${item.status}'. Must be one of: ${STATUSES.join(", ")}.`)
    }
    return { id: item.id!, content: item.content!, status: item.status as PlanStatus }
  })
}

export function formatPlan(items: readonly PlanItem[]): string {
  if (items.length === 0) return "Plan is empty."
  const done = items.filter((i) => i.status === "completed").length
  return [`Plan (${done}/${items.length} completed):`, ...items.map((i) => `${MARK[i.status]} ${i.content}`)].join("\n")
}
