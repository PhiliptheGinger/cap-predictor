"""Connector for the GitHub API.

This module provides helper functions to retrieve basic repository
metadata from the public GitHub API.  Only a small subset of fields is
persisted to disk to keep things lightweight and reduce noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import requests
from loguru import logger

GITHUB_API_URL = "https://api.github.com/repos/{owner}/{repo}"


def fetch_repo(owner: str, repo: str, token: str | None = None) -> Dict[str, Any]:
    """Fetch metadata for a GitHub repository.

    Parameters
    ----------
    owner:
        Repository owner.
    repo:
        Repository name.
    token:
        Optional personal access token to increase rate limits.
    """

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = GITHUB_API_URL.format(owner=owner, repo=repo)
    logger.debug("Querying GitHub API: %s", url)
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def update_store(
    path: Path, owner: str, repo: str, token: str | None = None
) -> Path:
    """Fetch repository information and persist it to *path*.

    Only the most relevant metadata fields are written to the JSON file.
    Returns the *path* written to for convenience.
    """

    data = fetch_repo(owner, repo, token)
    selected = {
        "full_name": data.get("full_name"),
        "description": data.get("description"),
        "html_url": data.get("html_url"),
        "stargazers_count": data.get("stargazers_count"),
        "forks_count": data.get("forks_count"),
        "open_issues_count": data.get("open_issues_count"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selected, indent=2))
    logger.info("Saved GitHub repo %s to %s", selected.get("full_name"), path)
    return path


__all__ = ["fetch_repo", "update_store"]
