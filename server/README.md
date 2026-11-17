# ml-intern-mcp

MCP server with machine learning research tools and GPU compute tools, used by the `opencode-ml-intern` plugin and usable from any MCP client.

## Research tools

Ported from [ml-intern](https://github.com/huggingface/ml-intern) (Apache-2.0):

| Tool | What it does |
|---|---|
| `hf_papers` | Searches papers, reads sections, finds linked models, datasets and collections, citation graph |
| `explore_hf_docs` | Lists the pages of an HF library documentation |
| `fetch_hf_docs` | Reads an HF documentation page as markdown |
| `find_hf_api` | Finds HF Hub REST endpoints with curl examples |
| `hf_inspect_dataset` | Shows dataset configs, splits, schema and sample rows |
| `hf_repo_files` | Lists, reads, uploads and deletes files in HF repos |
| `github_find_examples` | Finds example files in a GitHub repository |
| `github_list_repos` | Lists the repositories of a GitHub user or organization |
| `github_read_file` | Reads a file from a GitHub repository |

Tokens: `HF_TOKEN` or the cached `hf auth login` token; `GITHUB_TOKEN` or the `gh auth token` of the GitHub CLI.

## Running

```sh
uv run ml-intern-mcp            # stdio server
uv run ml-intern-mcp --fresh    # clear the result cache first
uv run ml-intern-mcp --no-cache # disable the result cache
```

Successful results of read-only tools are cached on disk for 6 hours in `$XDG_CACHE_HOME/ml-intern-mcp`.

## Development

```sh
uv run pytest
uv run python scripts/smoke_research.py   # live calls to every research tool
```
