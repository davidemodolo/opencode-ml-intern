from mcp import Client

from ml_intern_mcp import __version__
from ml_intern_mcp.server import build_server


async def test_server_info():
    async with Client(build_server()) as client:
        result = await client.call_tool("server_info", {})
    assert result.structured_content == {"name": "ml-intern-mcp", "version": __version__}
