import pytest

from ml_intern_mcp.tools import github_list_repos


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
