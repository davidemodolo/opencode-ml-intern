from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server.mcpserver.tools.base import Tool
from mcp.types import CallToolResult, TextContent
from pydantic import ConfigDict, Field

from ml_intern_mcp.cache import ResultCache

Handler = Callable[[dict[str, Any]], Awaitable[tuple[str, bool]]]


def _placeholder(arguments: dict[str, Any]) -> None:
    return None


class SpecTool(Tool):
    """A tool defined by a JSON schema spec and a handler returning (text, success)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    handler: Handler = Field(exclude=True)
    cache: ResultCache | None = Field(default=None, exclude=True)

    async def run(
        self, arguments: dict[str, Any], context: Any, convert_result: bool = False
    ) -> Any:
        arguments = arguments or {}
        cached = self.cache.get(self.name, arguments) if self.cache else None
        if cached is not None:
            return _result(cached, ok=True)
        text, ok = await self.handler(arguments)
        if ok and self.cache:
            self.cache.put(self.name, arguments, text)
        return _result(text, ok)


def _result(text: str, ok: bool) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=not ok)


def spec_tool(spec: dict[str, Any], handler: Handler, cache: ResultCache | None = None) -> SpecTool:
    base = Tool.from_function(
        _placeholder, name=spec["name"], description=spec["description"], structured_output=False
    )
    return SpecTool(
        fn=_placeholder,
        handler=handler,
        cache=cache,
        name=spec["name"],
        description=spec["description"],
        parameters=spec["parameters"],
        fn_metadata=base.fn_metadata,
        is_async=True,
    )
