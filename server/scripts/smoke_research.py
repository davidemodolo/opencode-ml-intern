"""Calls each research tool once against the live services and prints a short summary."""

import asyncio
import sys

from mcp import Client

from ml_intern_mcp.research import ensure_github_token, load_api_search_spec, research_tools
from ml_intern_mcp.server import build_server

CALLS = [
    ("hf_papers", {"operation": "search", "query": "LoRA fine-tuning", "limit": 2}),
    ("explore_hf_docs", {"endpoint": "trl"}),
    ("fetch_hf_docs", {"url": "https://huggingface.co/docs/trl/sft_trainer"}),
    ("hf_inspect_dataset", {"dataset": "roneneldan/TinyStories"}),
    ("hf_repo_files", {"operation": "list", "repo_id": "Qwen/Qwen2.5-0.5B"}),
    ("github_find_examples", {"repo": "trl", "org": "huggingface", "keyword": "sft"}),
    ("github_list_repos", {"owner": "huggingface", "limit": 3}),
    ("github_read_file", {"repo": "huggingface/trl", "path": "README.md"}),
    ("find_hf_api", {"query": "space logs"}),
]


async def main() -> int:
    ensure_github_token()
    tools = research_tools(api_search_spec=await load_api_search_spec())
    failures = 0
    async with Client(build_server(tools)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        for name, arguments in CALLS:
            if name not in names:
                print(f"SKIP {name}: not registered")
                continue
            result = await client.call_tool(name, arguments)
            text = result.content[0].text
            status = "FAIL" if result.is_error else "OK"
            failures += result.is_error
            print(f"{status} {name}: {text[:160]!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
