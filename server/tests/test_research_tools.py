import asyncio
import time

import httpx
import pytest
import requests

from ml_intern_mcp.tools import dataset_tools, github_list_repos
from ml_intern_mcp.tools.github_find_examples import github_find_examples_handler
from ml_intern_mcp.tools.github_read_file import github_read_file_handler, read_file
from ml_intern_mcp.tools.hf_repo_files_tool import HfRepoFilesTool


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self.payload


def repo(index):
    return {"name": f"r{index}", "full_name": f"o/r{index}", "stargazers_count": index}


@pytest.fixture
def github_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "t")


@pytest.fixture
def two_pages_of_repos(monkeypatch, github_token):
    requested = []
    pages = {1: [repo(i) for i in range(100)], 2: [repo(i) for i in range(100, 150)]}

    def fake_get(url, headers=None, params=None, timeout=None):
        requested.append((url, params))
        items = pages.get(params["page"], [])
        return FakeResponse({"items": items} if "search" in url else items)

    monkeypatch.setattr(github_list_repos.requests, "get", fake_get)
    return requested


def test_list_repos_lets_github_sort_by_stars(two_pages_of_repos):
    github_list_repos.list_repos("o", sort="stars", limit=3)
    url, params = two_pages_of_repos[0]
    assert url == "https://api.github.com/search/repositories" and len(two_pages_of_repos) == 1
    assert params["q"] == "org:o fork:true" and params["sort"] == "stars"


async def test_list_repos_handler_defaults_to_30_results(two_pages_of_repos):
    text, ok = await github_list_repos.github_list_repos_handler({"owner": "o"})
    assert ok and text.startswith("**Found 30 repositories")


async def test_github_handlers_do_not_block_the_event_loop(monkeypatch, github_token):
    def slow_get(*args, **kwargs):
        time.sleep(0.5)
        return FakeResponse({}, status_code=500)

    monkeypatch.setattr(requests, "get", slow_get)
    start = time.monotonic()
    await asyncio.gather(
        github_list_repos.github_list_repos_handler({"owner": "o"}),
        github_read_file_handler({"repo": "o/r", "path": "a.py"}),
        github_find_examples_handler({"repo": "r"}),
    )
    assert time.monotonic() - start < 1.0


async def test_hf_repo_listing_pages_off_the_event_loop():
    class SlowTreeApi:
        def list_repo_tree(self, **kwargs):
            time.sleep(0.5)
            yield from []

    tool = HfRepoFilesTool()
    tool.api = SlowTreeApi()
    start = time.monotonic()
    await asyncio.gather(*(tool.execute({"operation": "list", "repo_id": "r"}) for _ in range(3)))
    assert time.monotonic() - start < 1.0


def test_read_file_escapes_the_path(monkeypatch, github_token):
    urls = []

    def fake_get(url, **kwargs):
        urls.append(url)
        return FakeResponse({}, status_code=404)

    monkeypatch.setattr(requests, "get", fake_get)
    read_file("o/r", "notebooks/Fine-tune #2?.ipynb")
    assert urls == ["https://api.github.com/repos/o/r/contents/notebooks/Fine-tune%20%232%3F.ipynb"]


def mock_datasets_server(monkeypatch, respond):
    requested = []

    def handler(request):
        requested.append((request.url.path, dict(request.url.params)))
        status, payload = respond(request.url.path)
        return httpx.Response(status, json=payload)

    real_client = httpx.AsyncClient

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(dataset_tools.httpx, "AsyncClient", client)
    return requested


async def test_inspect_dataset_reports_server_errors(monkeypatch):
    mock_datasets_server(monkeypatch, lambda path: (404, {"error": "The dataset does not exist."}))
    text, ok = await dataset_tools.hf_inspect_dataset_handler({"dataset": "o/missing"})
    assert not ok and "The dataset does not exist." in text
