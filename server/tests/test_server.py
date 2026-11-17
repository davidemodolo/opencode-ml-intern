from mcp import Client

from ml_intern_mcp import __version__
from ml_intern_mcp.research import research_tools
from ml_intern_mcp.server import build_server


async def test_server_info():
    async with Client(build_server([])) as client:
        result = await client.call_tool("server_info", {})
    assert result.structured_content == {"name": "ml-intern-mcp", "version": __version__}


async def test_research_tools_are_listed_with_their_specs():
    tools = research_tools()
    async with Client(build_server(tools)) as client:
        listed = {tool.name: tool for tool in (await client.list_tools()).tools}
    for tool in tools:
        assert listed[tool.name].input_schema == tool.parameters
    assert {"hf_papers", "hf_inspect_dataset", "github_read_file", "hf_repo_write"} <= listed.keys()
