import asyncio
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import partial
from typing import Any

import httpx
from huggingface_hub import get_token

from ml_intern_mcp.cache import ResultCache
from ml_intern_mcp.spec_tool import SpecTool, spec_tool
from ml_intern_mcp.tools import docs_tools
from ml_intern_mcp.tools.dataset_tools import (
    HF_INSPECT_DATASET_TOOL_SPEC,
    hf_inspect_dataset_handler,
)
from ml_intern_mcp.tools.github_find_examples import (
    GITHUB_FIND_EXAMPLES_TOOL_SPEC,
    github_find_examples_handler,
)
from ml_intern_mcp.tools.github_list_repos import (
    GITHUB_LIST_REPOS_TOOL_SPEC,
    github_list_repos_handler,
)
from ml_intern_mcp.tools.github_read_file import (
    GITHUB_READ_FILE_TOOL_SPEC,
    github_read_file_handler,
)
from ml_intern_mcp.tools.hf_repo_files_tool import (
    HF_REPO_FILES_TOOL_SPEC,
    HF_REPO_WRITE_TOOL_SPEC,
    hf_repo_files_handler,
    hf_repo_write_handler,
)
from ml_intern_mcp.tools.papers_tool import HF_PAPERS_TOOL_SPEC, hf_papers_handler


@dataclass
class HubSession:
    hf_token: str | None


def ensure_github_token() -> None:
    if os.environ.get("GITHUB_TOKEN") or not shutil.which("gh"):
        return
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=False)
    if result.returncode == 0 and result.stdout.strip():
        os.environ["GITHUB_TOKEN"] = result.stdout.strip()


def research_tools(
    cache: ResultCache | None = None, api_search_spec: dict[str, Any] | None = None
) -> list[SpecTool]:
    session = HubSession(hf_token=get_token())
    tools = [
        spec_tool(HF_PAPERS_TOOL_SPEC, hf_papers_handler, cache),
        spec_tool(
            docs_tools.EXPLORE_HF_DOCS_TOOL_SPEC,
            partial(docs_tools.explore_hf_docs_handler, session=session),
            cache,
        ),
        spec_tool(
            docs_tools.HF_DOCS_FETCH_TOOL_SPEC,
            partial(docs_tools.hf_docs_fetch_handler, session=session),
            cache,
        ),
        spec_tool(
            HF_INSPECT_DATASET_TOOL_SPEC,
            partial(hf_inspect_dataset_handler, session=session),
            cache,
        ),
        spec_tool(HF_REPO_FILES_TOOL_SPEC, partial(hf_repo_files_handler, session=session)),
        spec_tool(HF_REPO_WRITE_TOOL_SPEC, partial(hf_repo_write_handler, session=session)),
        spec_tool(GITHUB_FIND_EXAMPLES_TOOL_SPEC, github_find_examples_handler, cache),
        spec_tool(GITHUB_LIST_REPOS_TOOL_SPEC, github_list_repos_handler, cache),
        spec_tool(GITHUB_READ_FILE_TOOL_SPEC, github_read_file_handler, cache),
    ]
    if api_search_spec is not None:
        tools.append(spec_tool(api_search_spec, docs_tools.search_openapi_handler, cache))
    return tools


async def load_api_search_spec() -> dict[str, Any] | None:
    try:
        return await asyncio.wait_for(docs_tools._get_api_search_tool_spec(), timeout=20)
    except (httpx.HTTPError, TimeoutError, ValueError, KeyError):
        return None
