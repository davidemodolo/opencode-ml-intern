import argparse
import asyncio
import logging
import os
import shlex
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.tools.base import Tool

from ml_intern_mcp import __version__
from ml_intern_mcp.cache import ResultCache
from ml_intern_mcp.compute.local import LocalBackend
from ml_intern_mcp.compute.service import ComputeService
from ml_intern_mcp.compute.store import JobStore
from ml_intern_mcp.compute.tools import register_compute_tools
from ml_intern_mcp.paths import cache_dir, data_dir
from ml_intern_mcp.research import ensure_github_token, load_api_search_spec, research_tools

INSTRUCTIONS = "Research and GPU compute tools for machine learning work."
CACHE_TTL_SECONDS = 6 * 3600
RECONCILE_SECONDS = 2.0
DEFAULT_RUNNER = "uv run --script"


def _lifespan(service: ComputeService | None):
    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
        if service is None:
            yield None
            return

        async def reconcile_forever() -> None:
            while True:
                await asyncio.to_thread(service.reconcile)
                await asyncio.sleep(RECONCILE_SECONDS)

        task = asyncio.create_task(reconcile_forever())
        try:
            yield None
        finally:
            task.cancel()

    return lifespan


def build_server(tools: list[Tool], service: ComputeService | None = None) -> MCPServer:
    server = MCPServer(
        name="ml-intern",
        version=__version__,
        instructions=INSTRUCTIONS,
        tools=tools,
        lifespan=_lifespan(service),
    )

    @server.tool(description="Return the server name and version.")
    def server_info() -> dict[str, str]:
        return {"name": "ml-intern-mcp", "version": __version__}

    if service is not None:
        register_compute_tools(server, service)
    return server


def build_compute_service() -> ComputeService:
    store = JobStore(data_dir())
    runner = shlex.split(os.environ.get("ML_INTERN_JOB_RUNNER", DEFAULT_RUNNER))
    return ComputeService(store, {"local": LocalBackend(store, runner)})


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
    service = build_compute_service()
    service.reconcile()
    build_server(research_tools(cache, api_search_spec), service).run()
