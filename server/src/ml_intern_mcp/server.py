from mcp.server.mcpserver import MCPServer

from ml_intern_mcp import __version__

INSTRUCTIONS = "Research and GPU compute tools for machine learning work."


def build_server() -> MCPServer:
    server = MCPServer(name="ml-intern", version=__version__, instructions=INSTRUCTIONS)

    @server.tool(description="Return the server name and version.")
    def server_info() -> dict[str, str]:
        return {"name": "ml-intern-mcp", "version": __version__}

    return server


def main() -> None:
    build_server().run()
