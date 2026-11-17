import type { Options } from "./options.js"

export interface McpEditorLike {
  get(name: string): unknown
  set(name: string, config: any): void
}

/** Registers the ml-intern MCP server unless the user configured a server with the same name. */
export function registerServer(editor: McpEditorLike, options: Options): boolean {
  if (editor.get(options.mcpServer) !== undefined) return false
  editor.set(options.mcpServer, {
    type: "local",
    command: [...options.serverCommand],
    environment: { ML_INTERN_DATA_DIR: options.dataDir },
  })
  return true
}
