from mcp import Client

from ml_intern_mcp.cache import ResultCache
from ml_intern_mcp.server import build_server
from ml_intern_mcp.spec_tool import spec_tool

SPEC = {
    "name": "echo",
    "description": "Echo the text",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


def counting_handler(calls: list):
    async def handler(arguments):
        calls.append(arguments)
        if arguments["text"] == "fail":
            return "Error: failed", False
        return "echo: " + arguments["text"], True

    return handler


async def test_spec_tool_returns_text_and_error_flag():
    calls: list = []
    tool = spec_tool(SPEC, counting_handler(calls))
    async with Client(build_server([tool])) as client:
        ok = await client.call_tool("echo", {"text": "hi"})
        bad = await client.call_tool("echo", {"text": "fail"})
    assert ok.content[0].text == "echo: hi" and not ok.is_error
    assert bad.content[0].text == "Error: failed" and bad.is_error


async def test_cache_serves_repeated_calls_and_skips_errors(tmp_path):
    calls: list = []
    cache = ResultCache(tmp_path, ttl_seconds=60)
    tool = spec_tool(SPEC, counting_handler(calls), cache)
    async with Client(build_server([tool])) as client:
        await client.call_tool("echo", {"text": "hi"})
        again = await client.call_tool("echo", {"text": "hi"})
        await client.call_tool("echo", {"text": "fail"})
        await client.call_tool("echo", {"text": "fail"})
    assert again.content[0].text == "echo: hi"
    assert [c["text"] for c in calls] == ["hi", "fail", "fail"]


async def test_cache_expires(tmp_path):
    calls: list = []
    cache = ResultCache(tmp_path, ttl_seconds=-1)
    tool = spec_tool(SPEC, counting_handler(calls), cache)
    async with Client(build_server([tool])) as client:
        await client.call_tool("echo", {"text": "hi"})
        await client.call_tool("echo", {"text": "hi"})
    assert len(calls) == 2
