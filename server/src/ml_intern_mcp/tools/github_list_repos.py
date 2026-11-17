"""
GitHub List Repositories Tool - List and sort repositories for any user or organization

Efficiently discover repositories with flexible sorting options.
"""

from typing import Any, Dict, Literal, Optional

import requests

from ml_intern_mcp.tools._github import (
    github_error,
    github_headers,
    github_token,
    missing_token,
    short_description,
)
from ml_intern_mcp.tools.types import ToolResult, error_result, handler_output


def list_repos(
    owner: str,
    owner_type: Literal["user", "org"] = "org",
    sort: Literal["stars", "forks", "updated", "created"] = "stars",
    order: Literal["asc", "desc"] = "desc",
    limit: Optional[int] = 30,
) -> ToolResult:
    """
    List repositories for a user or organization using GitHub REST API.

    Args:
        owner: GitHub username or organization name
        owner_type: Whether the owner is a "user" or "org" (default: "org")
        sort: Sort field - "stars", "forks", "updated", or "created"
        order: Sort order - "asc" or "desc" (default: "desc")
        limit: Maximum number of repositories to return

    Returns:
        ToolResult with repository information
    """
    token = github_token()
    if not token:
        return missing_token()

    headers = github_headers(token)
    per_page = 100  # Maximum allowed by GitHub

    # The list API cannot sort by stars or forks, the search API can
    if sort in ("stars", "forks"):
        qualifier = "org" if owner_type == "org" else "user"
        url = "https://api.github.com/search/repositories"
        base_params = {"q": f"{qualifier}:{owner} fork:true", "sort": sort, "order": order}
    else:
        url = f"https://api.github.com/{'orgs' if owner_type == 'org' else 'users'}/{owner}/repos"
        base_params = {"sort": sort, "direction": order}

    all_repos = []
    page = 1
    try:
        while True:
            params = {**base_params, "page": page, "per_page": per_page}
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=30,
            )

            if response.status_code == 403:
                error_data = response.json()
                return error_result(f"GitHub API rate limit or permission error: {error_data.get('message', 'Unknown error')}")

            if response.status_code != 200:
                return error_result(github_error(response))

            data = response.json()
            items = data.get("items", []) if isinstance(data, dict) else data

            if not items:
                break

            for item in items:
                all_repos.append(
                    {
                        "name": item.get("name"),
                        "full_name": item.get("full_name"),
                        "description": item.get("description"),
                        "html_url": item.get("html_url"),
                        "language": item.get("language"),
                        "stars": item.get("stargazers_count", 0),
                        "forks": item.get("forks_count", 0),
                        "open_issues": item.get("open_issues_count", 0),
                        "topics": item.get("topics", []),
                        "updated_at": item.get("updated_at"),
                        "created_at": item.get("created_at"),
                    }
                )

            # Check if we got fewer results than requested (last page)
            if len(items) < per_page:
                break

            # Stop if we have enough repos
            if limit and len(all_repos) >= limit:
                break

            page += 1

    except requests.exceptions.RequestException as e:
        return error_result(f"Failed to connect to GitHub API: {str(e)}")

    # Apply limit after sorting
    if limit:
        all_repos = all_repos[:limit]

    if not all_repos:
        return {
            "formatted": f"No repositories found for {owner_type} '{owner}'",
            "totalResults": 0,
            "resultsShared": 0,
        }

    # Format output
    lines = [f"**Found {len(all_repos)} repositories for {owner}:**\n"]

    for i, repo in enumerate(all_repos, 1):
        lines.append(f"{i}. **{repo['full_name']}**")
        lines.append(
            f"   ⭐ {repo['stars']:,} stars | 🍴 {repo['forks']:,} forks | Language: {repo['language'] or 'N/A'}"
        )
        if repo["description"]:
            desc = short_description(repo["description"])
            lines.append(f"   {desc}")
        lines.append(f"   URL: {repo['html_url']}")
        if repo["topics"]:
            lines.append(f"   Topics: {', '.join(repo['topics'][:5])}")

        # Copyable parameters for other tools
        lines.append(f"   Use in tools: {{'repo': '{repo['full_name']}'}}")
        lines.append("")

    return {
        "formatted": "\n".join(lines),
        "totalResults": len(all_repos),
        "resultsShared": len(all_repos),
    }


# Tool specification
GITHUB_LIST_REPOS_TOOL_SPEC = {
    "name": "github_list_repos",
    "description": (
        "List and discover repositories for GitHub organizations or users with flexible sorting. "
        "**Use when:** (1) Exploring what libraries exist for a task, (2) Finding the right library to use, "
        "(3) Discovering popular or active projects, (4) Checking recently updated repos for latest features, "
        "(5) Finding alternative libraries in an organization. "
        "**Pattern:** github_list_repos (discover libraries) → github_find_examples (find usage examples) → implement. "
        "Returns: Comprehensive repository information (stars, forks, language, topics, URLs), sorted by preference. "
        "**Then:** Use github_find_examples on selected repo to discover example code. "
        "Sorts by: stars (popularity), forks (community), updated (activity), created (age).\n\n"
        "## When to use this tool\n\n"
        "- When you need to find libraries to use in your implementation\n"
        "- When exploring what repositories exist for a task or domain\n"
        "- When debugging an error and looking up if others have similar issues in repos\n"
        "- When finding the most popular or actively maintained projects for a user/org\n"
        "## Examples\n\n"
        "<example>\n"
        "// ML Workflow Step: Discover HF libraries for RLHF/alignment\n"
        "// Use case: Find the right library for training with human feedback\n"
        "{\n"
        "  owner: 'huggingface',\n"
        "  owner_type: 'org',\n"
        "  sort: 'stars',\n"
        "  limit: 10\n"
        "}\n"
        "// Returns: transformers, trl, peft, accelerate, diffusers...\n"
        "</example>\n\n"
        "<example>\n"
        "// ML Workflow Step: Check for recently updated HF repos\n"
        "// Use case: Find actively maintained libraries with latest features\n"
        "{\n"
        "  owner: 'huggingface',\n"
        "  owner_type: 'org',\n"
        "  sort: 'updated',\n"
        "  order: 'desc',\n"
        "  limit: 15\n"
        "}\n"
        "// Helps identify which repos have recent improvements/fixes\n"
        "</example>"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "owner": {
                "type": "string",
                "description": "GitHub username or organization name. Required.",
            },
            "owner_type": {
                "type": "string",
                "enum": ["user", "org"],
                "description": "Whether the owner is a 'user' or 'org'. Default: 'org'.",
            },
            "sort": {
                "type": "string",
                "enum": ["stars", "forks", "updated", "created"],
                "description": "Sort field. Options: 'stars', 'forks', 'updated', 'created'. Default: 'stars'.",
            },
            "order": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": "Sort order. Options: 'asc', 'desc'. Default: 'desc'.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of repositories to return. Default: 30.",
            },
        },
        "required": ["owner"],
    },
}


async def github_list_repos_handler(arguments: Dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router"""
    try:
        result = list_repos(
            owner=arguments["owner"],
            owner_type=arguments.get("owner_type", "org"),
            sort=arguments.get("sort", "stars"),
            order=arguments.get("order", "desc"),
            limit=arguments.get("limit", 30),
        )
        return handler_output(result)
    except Exception as e:
        return f"Error listing repositories: {str(e)}", False
