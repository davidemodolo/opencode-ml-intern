"""
GitHub Read File Tool - Read file contents from any GitHub repository with line range support

Fetch exact file contents with metadata, supporting line ranges for efficient reading.
"""

import asyncio
import base64
import json
from typing import Any, Dict, Optional
from urllib.parse import quote

import nbformat
import requests
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import ClearOutputPreprocessor, TagRemovePreprocessor

from ml_intern_mcp.tools._github import (
    github_error,
    github_headers,
    github_token,
    missing_token,
)
from ml_intern_mcp.tools.types import ToolResult, error_result, handler_output


def _convert_ipynb_to_markdown(content: str) -> str:
    """
    Convert Jupyter notebook JSON to LLM-friendly Markdown.

    Args:
        content: Raw notebook JSON string

    Returns:
        Converted Markdown string
    """
    try:
        # Parse notebook JSON
        nb_dict = json.loads(content)

        # Normalize cell sources (can be string or list of strings)
        if "cells" in nb_dict:
            for cell in nb_dict["cells"]:
                if "source" in cell and isinstance(cell["source"], list):
                    cell["source"] = "".join(cell["source"])

        # Read notebook with explicit version
        nb = nbformat.reads(json.dumps(nb_dict), as_version=4)

        # Strip outputs for LLM readability (outputs can be noisy/large)
        clear = ClearOutputPreprocessor()
        nb, _ = clear.preprocess(nb, {})

        # Optionally remove cells tagged with "hide" or similar
        remove = TagRemovePreprocessor(
            remove_cell_tags={"hide", "hidden", "remove"},
            remove_input_tags=set(),
            remove_all_outputs_tags=set(),
        )
        nb, _ = remove.preprocess(nb, {})

        # Convert to markdown
        exporter = MarkdownExporter()
        markdown, _ = exporter.from_notebook_node(nb)

        return markdown

    except json.JSONDecodeError:
        return content
    except Exception:
        return content


def read_file(
    repo: str,
    path: str,
    ref: str = "HEAD",
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
) -> ToolResult:
    """
    Read file contents from a GitHub repository with line range support.

    Args:
        repo: Repository in format "owner/repo" (e.g., "github/github-mcp-server")
        path: Path to file in repository (e.g., "pkg/github/search.go")
        ref: Git reference - branch name, tag, or commit SHA (default: "HEAD")
        line_start: Starting line number (1-indexed, inclusive)
        line_end: Ending line number (1-indexed, inclusive)

    Returns:
        ToolResult with file contents and metadata
    """
    token = github_token()
    if not token:
        return missing_token()

    # Parse repo
    if "/" not in repo:
        return error_result("Error: repo must be in format 'owner/repo'")

    owner, repo_name = repo.split("/", 1)

    headers = github_headers(token)

    # Fetch file contents
    url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{quote(path, safe='/')}"
    params = {}
    if ref and ref != "HEAD":
        params["ref"] = ref

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)

        if response.status_code == 404:
            return error_result(f"File not found: {path} in {repo} (ref: {ref})")

        if response.status_code != 200:
            return error_result(github_error(response))

        data = response.json()

        # Check if it's a file
        if data.get("type") != "file":
            return error_result(f"Path {path} is not a file (type: {data.get('type')})")

        # Decode content
        content_b64 = data.get("content", "")
        if content_b64:
            content_b64 = content_b64.replace("\n", "").replace(" ", "")
            content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        else:
            # For large files, fetch raw content
            raw_headers = github_headers(token, accept="application/vnd.github.raw")
            raw_response = requests.get(
                url, headers=raw_headers, params=params, timeout=30
            )
            if raw_response.status_code != 200:
                return error_result("Failed to fetch file content")
            content = raw_response.text

        if path.lower().endswith(".ipynb"):
            content = _convert_ipynb_to_markdown(content)

        # Process line ranges
        lines = content.split("\n")
        total_lines = len(lines)

        truncated = False

        if line_start is None and line_end is None:
            # No range specified
            if total_lines > 300:
                line_start = 1
                line_end = 300
                truncated = True
            else:
                line_start = 1
                line_end = total_lines
        else:
            # Range specified
            if line_start is None:
                line_start = 1
            if line_end is None:
                line_end = total_lines

            # Validate range
            line_start = max(1, line_start)
            line_end = min(total_lines, line_end)
            if line_start > line_end:
                return error_result(f"Invalid range: line_start ({line_start}) > line_end ({line_end})")

        # Extract lines
        selected_lines = lines[line_start - 1 : line_end]
        selected_content = "\n".join(selected_lines)

        # Format output
        lines_output = [f"**Reading file from repo: {repo}, path: {path}**"]

        if ref and ref != "HEAD":
            lines_output.append(f"Ref: {ref}")

        lines_output.append("\n**File content:")
        lines_output.append("```")
        lines_output.append(selected_content)
        lines_output.append("```")
        if truncated:
            lines_output.append(
                f"Currently showing lines {line_start}-{line_end} out of {total_lines} total lines. Use line_start and line_end to view more lines."
            )
        return {
            "formatted": "\n".join(lines_output),
            "totalResults": 1,
            "resultsShared": 1,
        }

    except requests.exceptions.RequestException as e:
        return error_result(f"Failed to connect to GitHub API: {str(e)}")


# Tool specification
GITHUB_READ_FILE_TOOL_SPEC = {
    "name": "github_read_file",
    "description": (
        "Read file contents from GitHub repositories. Returns first 300 lines by default. "
        "Auto-converts Jupyter notebooks to markdown.\n\n"
        "Use AFTER github_find_examples to study the working implementation. "
        "The purpose is to learn current API patterns — imports, trainer configs, dataset handling — "
        "so your implementation uses correct, up-to-date code.\n\n"
        "Use line_start/line_end for large files (>300 lines) to read specific sections.\n\n"
        "When NOT to use: when you don't know the file path (use github_find_examples first)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository in format 'owner/repo' (e.g., 'github/github-mcp-server'). Required.",
            },
            "path": {
                "type": "string",
                "description": "Path to file in repository (e.g., 'src/index.js'). Required.",
            },
            "ref": {
                "type": "string",
                "description": "Git reference - branch name, tag, or commit SHA. Default: 'HEAD'.",
            },
            "line_start": {
                "type": "integer",
                "description": "Starting line number (1-indexed, inclusive). Optional.",
            },
            "line_end": {
                "type": "integer",
                "description": "Ending line number (1-indexed, inclusive). Optional.",
            },
        },
        "required": ["repo", "path"],
    },
}


async def github_read_file_handler(arguments: Dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router"""
    try:
        result = await asyncio.to_thread(
            read_file,
            repo=arguments["repo"],
            path=arguments["path"],
            ref=arguments.get("ref", "HEAD"),
            line_start=arguments.get("line_start"),
            line_end=arguments.get("line_end"),
        )
        return handler_output(result)
    except Exception as e:
        return f"Error reading file: {str(e)}", False
