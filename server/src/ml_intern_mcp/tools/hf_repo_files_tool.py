"""
HF Repo Files Tool - File operations on Hugging Face repositories

hf_repo_files lists and reads, hf_repo_write uploads and deletes.
"""

import asyncio
from typing import Any, Dict, Optional

import httpx
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from ml_intern_mcp.tools.types import ToolResult, error_result, handler_output

MAX_UTF8_BYTES = 4


async def _async_call(func, *args, **kwargs):
    """Wrap synchronous HfApi calls for async context."""
    return await asyncio.to_thread(func, *args, **kwargs)


def _build_repo_url(repo_id: str, repo_type: str = "model") -> str:
    """Build the Hub URL for a repository."""
    if repo_type == "model":
        return f"https://huggingface.co/{repo_id}"
    return f"https://huggingface.co/{repo_type}s/{repo_id}"


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


READ_OPERATIONS = ("list", "read")
WRITE_OPERATIONS = ("upload", "delete")

USAGE = {
    "list": '`{"operation": "list", "repo_id": "gpt2"}`',
    "read": '`{"operation": "read", "repo_id": "gpt2", "path": "config.json"}`',
    "upload": '`{"operation": "upload", "repo_id": "my-model", "path": "README.md", "content": "..."}`',
    "delete": '`{"operation": "delete", "repo_id": "my-model", "patterns": ["*.tmp"]}`',
}


class HfRepoFilesTool:
    """Tool for file operations on HF repos, limited to the given operations."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        name: str = "hf_repo_files",
        operations: tuple[str, ...] = READ_OPERATIONS,
    ):
        self.api = HfApi(token=hf_token)
        self.name = name
        self.operations = operations

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute the specified operation."""
        operation = args.get("operation")

        if not operation:
            return self._help()

        try:
            handlers = {
                "list": self._list,
                "read": self._read,
                "upload": self._upload,
                "delete": self._delete,
            }

            if operation in self.operations:
                return await handlers[operation](args)
            return error_result(
                f"Unknown operation: {operation}. Valid: {', '.join(self.operations)}"
            )

        except RepositoryNotFoundError:
            return error_result(f"Repository not found: {args.get('repo_id')}")
        except EntryNotFoundError:
            return error_result(f"File not found: {args.get('path')}")
        except Exception as e:
            return error_result(f"Error: {str(e)}")

    def _help(self) -> ToolResult:
        """Show usage instructions."""
        usage = "\n".join(f"- `{op}`: {USAGE[op]}" for op in self.operations)
        return {
            "formatted": f"**{self.name}** - File operations on HF repos\n\n**Operations:**\n{usage}\n\n"
            "**Common params:** repo_id (required), repo_type (model/dataset/space), revision (default: main)",
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _list(self, args: Dict[str, Any]) -> ToolResult:
        """List files in a repository."""
        repo_id = args.get("repo_id")
        if not repo_id:
            return error_result("repo_id is required")

        repo_type = args.get("repo_type", "model")
        revision = args.get("revision", "main")
        path = args.get("path", "")

        items = await _async_call(
            lambda: list(
                self.api.list_repo_tree(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    path_in_repo=path,
                    recursive=True,
                )
            )
        )

        if not items:
            return {
                "formatted": f"No files in {repo_id}",
                "totalResults": 0,
                "resultsShared": 0,
            }

        lines = []
        total_size = 0
        file_count = 0
        for item in sorted(items, key=lambda x: x.path):
            if isinstance(item, RepoFile):
                file_count += 1
                total_size += item.size
                lines.append(f"{item.path} ({_format_size(item.size)})")
            else:
                lines.append(f"{item.path}/")

        url = _build_repo_url(repo_id, repo_type)
        response = (
            f"**{repo_id}** ({file_count} files, {_format_size(total_size)})\n{url}/tree/{revision}\n\n"
            + "\n".join(lines)
        )

        return {
            "formatted": response,
            "totalResults": len(items),
            "resultsShared": len(items),
        }

    async def _read(self, args: Dict[str, Any]) -> ToolResult:
        """Read file content from a repository."""
        repo_id = args.get("repo_id")
        path = args.get("path")

        if not repo_id:
            return error_result("repo_id is required")
        if not path:
            return error_result("path is required")

        repo_type = args.get("repo_type", "model")
        revision = args.get("revision", "main")
        max_chars = args.get("max_chars", 50000)

        url = hf_hub_url(repo_id, path, repo_type=repo_type, revision=revision)
        headers = {"Range": f"bytes=0-{max_chars * MAX_UTF8_BYTES - 1}"}
        if self.api.token:
            headers["Authorization"] = f"Bearer {self.api.token}"
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
        if response.status_code == 404:
            return error_result(f"File not found: {path}")
        response.raise_for_status()

        data = response.content
        total = int(response.headers.get("content-range", "").rpartition("/")[2] or len(data))
        if b"\x00" in data:
            return {
                "formatted": f"Binary file ({_format_size(total)})",
                "totalResults": 1,
                "resultsShared": 1,
            }

        content = data.decode("utf-8", errors="replace")
        truncated = total > len(data) or len(content) > max_chars
        content = content[:max_chars]
        blob_url = f"{_build_repo_url(repo_id, repo_type)}/blob/{revision}/{path}"
        response_text = (
            f"**{path}**{' (truncated)' if truncated else ''}\n{blob_url}\n\n```\n{content}\n```"
        )
        return {"formatted": response_text, "totalResults": 1, "resultsShared": 1}

    async def _upload(self, args: Dict[str, Any]) -> ToolResult:
        """Upload content to a repository."""
        repo_id = args.get("repo_id")
        path = args.get("path")
        content = args.get("content")

        if not repo_id:
            return error_result("repo_id is required")
        if not path:
            return error_result("path is required")
        if content is None:
            return error_result("content is required")

        repo_type = args.get("repo_type", "model")
        revision = args.get("revision", "main")
        create_pr = args.get("create_pr", False)
        commit_message = args.get("commit_message", f"Upload {path}")

        file_bytes = content.encode("utf-8") if isinstance(content, str) else content

        result = await _async_call(
            self.api.upload_file,
            path_or_fileobj=file_bytes,
            path_in_repo=path,
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            commit_message=commit_message,
            create_pr=create_pr,
        )

        url = _build_repo_url(repo_id, repo_type)
        if create_pr and hasattr(result, "pr_url"):
            response = f"**Uploaded as PR**\n{result.pr_url}"
        else:
            response = f"**Uploaded:** {path}\n{url}/blob/{revision}/{path}"

        return {"formatted": response, "totalResults": 1, "resultsShared": 1}

    async def _delete(self, args: Dict[str, Any]) -> ToolResult:
        """Delete files from a repository."""
        repo_id = args.get("repo_id")
        patterns = args.get("patterns")

        if not repo_id:
            return error_result("repo_id is required")
        if not patterns:
            return error_result("patterns is required (list of paths/wildcards)")

        if isinstance(patterns, str):
            patterns = [patterns]

        repo_type = args.get("repo_type", "model")
        revision = args.get("revision", "main")
        create_pr = args.get("create_pr", False)
        commit_message = args.get("commit_message", f"Delete {', '.join(patterns)}")

        await _async_call(
            self.api.delete_files,
            repo_id=repo_id,
            delete_patterns=patterns,
            repo_type=repo_type,
            revision=revision,
            commit_message=commit_message,
            create_pr=create_pr,
        )

        response = f"**Deleted:** {', '.join(patterns)} from {repo_id}"
        return {"formatted": response, "totalResults": 1, "resultsShared": 1}


REPO_PROPERTIES = {
    "repo_id": {
        "type": "string",
        "description": "Repository ID (e.g., 'username/repo-name')",
    },
    "repo_type": {
        "type": "string",
        "enum": ["model", "dataset", "space"],
        "description": "Repository type (default: model)",
    },
    "revision": {
        "type": "string",
        "description": "Branch/tag/commit (default: main)",
    },
}

HF_REPO_FILES_TOOL_SPEC = {
    "name": "hf_repo_files",
    "description": (
        "List and read files in HF repos (models/datasets/spaces).\n\n"
        "## Operations\n"
        "- **list**: List files with sizes and structure\n"
        "- **read**: Read file content (text files only)\n\n"
        "## Use when\n"
        "- Need to see what files exist in a repo\n"
        "- Want to read config.json, README.md, or other text files\n\n"
        "## Examples\n"
        '{"operation": "list", "repo_id": "meta-llama/Llama-2-7b"}\n'
        '{"operation": "read", "repo_id": "gpt2", "path": "config.json"}\n\n'
        "## Notes\n"
        "- For binary files (safetensors, bin), use list to see them but can't read content\n"
        "- To change files in a repo, use hf_repo_write\n"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": list(READ_OPERATIONS),
                "description": "Operation: list, read",
            },
            **REPO_PROPERTIES,
            "path": {
                "type": "string",
                "description": "Folder to list or file to read",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters returned by read (default: 50000)",
            },
        },
        "required": ["operation"],
    },
}

HF_REPO_WRITE_TOOL_SPEC = {
    "name": "hf_repo_write",
    "description": (
        "Upload or delete files in HF repos (models/datasets/spaces). "
        "Changes the user's repos, so the user is asked to approve each call.\n\n"
        "## Operations\n"
        "- **upload**: Upload content to repo (can create PR)\n"
        "- **delete**: Delete files/folders (supports wildcards like *.tmp)\n\n"
        "## Use when\n"
        "- Uploading training scripts, configs, or results to a repo\n"
        "- Cleaning up temporary files from a repo\n\n"
        "## Examples\n"
        '{"operation": "upload", "repo_id": "my-model", "path": "README.md", "content": "# My Model"}\n'
        '{"operation": "upload", "repo_id": "org/model", "path": "fix.py", "content": "...", "create_pr": true}\n'
        '{"operation": "delete", "repo_id": "my-model", "patterns": ["*.tmp", "logs/"]}\n\n'
        "## Notes\n"
        "- Use create_pr=true to propose changes instead of direct commit\n"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": list(WRITE_OPERATIONS),
                "description": "Operation: upload, delete",
            },
            **REPO_PROPERTIES,
            "path": {
                "type": "string",
                "description": "File path for upload",
            },
            "content": {
                "type": "string",
                "description": "File content for upload",
            },
            "patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Patterns to delete (e.g., ['*.tmp', 'logs/'])",
            },
            "create_pr": {
                "type": "boolean",
                "description": "Create PR instead of direct commit",
            },
            "commit_message": {
                "type": "string",
                "description": "Custom commit message",
            },
        },
        "required": ["operation"],
    },
}


async def _run(arguments: Dict[str, Any], session, name: str, operations: tuple[str, ...]):
    try:
        hf_token = session.hf_token if session else None
        tool = HfRepoFilesTool(hf_token=hf_token, name=name, operations=operations)
        return handler_output(await tool.execute(arguments))
    except Exception as e:
        return f"Error: {str(e)}", False


async def hf_repo_files_handler(arguments: Dict[str, Any], session=None) -> tuple[str, bool]:
    """Handler for the read-only hf_repo_files tool."""
    return await _run(arguments, session, "hf_repo_files", READ_OPERATIONS)


async def hf_repo_write_handler(arguments: Dict[str, Any], session=None) -> tuple[str, bool]:
    """Handler for the hf_repo_write tool."""
    return await _run(arguments, session, "hf_repo_write", WRITE_OPERATIONS)
