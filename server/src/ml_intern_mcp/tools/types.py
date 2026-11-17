"""
Types for Hugging Face tools

Ported from: hf-mcp-server/packages/mcp/src/types/
"""

from typing import TypedDict


class ToolResult(TypedDict, total=False):
    """Result returned by HF tool operations"""

    formatted: str
    totalResults: int
    resultsShared: int
    isError: bool


def error_result(message: str) -> ToolResult:
    return {"formatted": message, "totalResults": 0, "resultsShared": 0, "isError": True}


def handler_output(result: ToolResult) -> tuple[str, bool]:
    return result["formatted"], not result.get("isError", False)
