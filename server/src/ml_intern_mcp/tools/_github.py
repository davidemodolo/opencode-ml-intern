"""Shared helpers for the GitHub tools."""

import os

import requests

from ml_intern_mcp.tools.types import ToolResult, error_result

API_VERSION = "2022-11-28"
MISSING_TOKEN = "Error: GITHUB_TOKEN environment variable is required"


def github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN")


def missing_token() -> ToolResult:
    return error_result(MISSING_TOKEN)


def github_headers(token: str, accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Accept": accept,
        "X-GitHub-Api-Version": API_VERSION,
        "Authorization": f"Bearer {token}",
    }


def short_description(text: str, limit: int = 100) -> str:
    return text[:limit] + "..." if len(text) > limit else text


def github_error(response: requests.Response) -> str:
    message = f"GitHub API error (status {response.status_code})"
    try:
        data = response.json()
    except ValueError:
        return message
    if isinstance(data, dict) and "message" in data:
        message += f": {data['message']}"
    return message
