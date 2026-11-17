import argparse
import asyncio
import logging
import shutil

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.tools.base import Tool

from ml_intern_mcp import __version__
from ml_intern_mcp.cache import ResultCache
from ml_intern_mcp.paths import cache_dir
from ml_intern_mcp.research import ensure_github_token, load_api_search_spec, research_tools

INSTRUCTIONS = "Research and GPU compute tools for machine learning work."
CACHE_TTL_SECONDS = 6 * 3600


def build_server(tools: list[Tool]) -> MCPServer:
    server = MCPServer(
        name="ml-intern", version=__version__, instructions=INSTRUCTIONS, tools=tools
    )

    @server.tool(description="Return the server name and version.")
    def server_info() -> dict[str, str]:
        return {"name": "ml-intern-mcp", "version": __version__}

    return server


def main() -> None:
    parser = argparse.ArgumentParser(prog="ml-intern-mcp")
    parser.add_argument("--fresh", action="store_true", help="clear the result cache on start")
    parser.add_argument("--no-cache", action="store_true", help="disable the result cache")
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root = cache_dir() / "results"
    if args.fresh:
        shutil.rmtree(root, ignore_errors=True)
    cache = ResultCache(root, CACHE_TTL_SECONDS, enabled=not args.no_cache)

    ensure_github_token()
    api_search_spec = asyncio.run(load_api_search_spec())
    build_server(research_tools(cache, api_search_spec)).run()
