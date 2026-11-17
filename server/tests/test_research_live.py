import os

import pytest
from mcp import Client

from ml_intern_mcp.research import ensure_github_token, load_api_search_spec, research_tools
from ml_intern_mcp.server import build_server

pytestmark = pytest.mark.skipif(
    os.environ.get("ML_INTERN_LIVE") != "1", reason="set ML_INTERN_LIVE=1 to call live services"
)

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


@pytest.mark.parametrize(("name", "arguments"), CALLS, ids=[c[0] for c in CALLS])
async def test_research_tool_live(name, arguments):
    ensure_github_token()
    tools = research_tools(api_search_spec=await load_api_search_spec())
    async with Client(build_server(tools)) as client:
        result = await client.call_tool(name, arguments)
    assert not result.is_error, result.content[0].text
    assert len(result.content[0].text) > 50
