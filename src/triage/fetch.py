"""Snapshot a repository's issues (and its label set) through the GitHub GraphQL API.

The snapshot is what the offline benchmark runs against: every issue's text, labels, state and,
for issues closed as duplicates, which issue they duplicate. Comments are deliberately left out:
on a duplicate they usually say "duplicate of #123", which would hand the agent the answer.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

GRAPHQL_URL = "https://api.github.com/graphql"
MAX_BODY_CHARS = 4000
DUPLICATE_COMMENT = re.compile(r"\b(?:duplicate|dupe) of #(\d+)", re.IGNORECASE)

ISSUES_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    issues(first: 100, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body state stateReason createdAt closedAt
        labels(first: 20) { nodes { name } }
        timelineItems(last: 3, itemTypes: [CLOSED_EVENT]) {
          nodes { ... on ClosedEvent { duplicateOf { ... on Issue { number } } } }
        }
        comments(last: 3) { nodes { body } }
      }
    }
  }
}
"""

LABELS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    labels(first: 100, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes { name description }
    }
  }
}
"""


def github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sys.exit("Set GITHUB_TOKEN, or log in with the GitHub CLI (`gh auth login`).")


class GraphQL:
    def __init__(self, token: str):
        self._client = httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=60)

    def query(self, query: str, **variables) -> dict:
        for attempt in range(5):
            r = self._client.post(GRAPHQL_URL, json={"query": query, "variables": variables})
            if r.status_code in (502, 503, 504):  # GitHub times out on heavy pages now and then
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            payload = r.json()
            if "errors" in payload:
                raise RuntimeError(payload["errors"])
            return payload["data"]
        r.raise_for_status()
        return {}


def _duplicate_of(node: dict) -> int | None:
    """The canonical issue for one closed as a duplicate: GitHub's own link, else a "Duplicate of #N" comment."""
    if node["stateReason"] != "DUPLICATE":
        return None
    for event in reversed(node["timelineItems"]["nodes"]):
        canonical = (event or {}).get("duplicateOf") or {}
        if canonical.get("number"):
            return canonical["number"]
    for comment in reversed(node["comments"]["nodes"]):
        m = DUPLICATE_COMMENT.search(comment["body"] or "")
        if m:
            return int(m.group(1))
    return None


def _issue_record(node: dict) -> dict:
    return {
        "number": node["number"],
        "title": node["title"],
        "body": (node["body"] or "")[:MAX_BODY_CHARS],
        "state": node["state"].lower(),
        "state_reason": (node["stateReason"] or "").lower() or None,
        "labels": [label["name"] for label in node["labels"]["nodes"]],
        "created_at": node["createdAt"],
        "closed_at": node["closedAt"],
        "duplicate_of": _duplicate_of(node),
    }


def fetch_snapshot(repo: str, out_dir: Path, limit: int) -> None:
    owner, name = repo.split("/")
    gql = GraphQL(github_token())
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, cursor = [], None
    while True:
        page = gql.query(LABELS_QUERY, owner=owner, name=name, cursor=cursor)["repository"]["labels"]
        labels += [{"name": n["name"], "description": n["description"] or ""} for n in page["nodes"]]
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")

    issues, cursor = [], None
    while len(issues) < limit:
        page = gql.query(ISSUES_QUERY, owner=owner, name=name, cursor=cursor)["repository"]["issues"]
        issues += [_issue_record(n) for n in page["nodes"]]
        print(f"\r{len(issues)} issues", end="", file=sys.stderr, flush=True)
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    print(file=sys.stderr)

    issues.sort(key=lambda i: i["number"])
    with gzip.open(out_dir / "issues.jsonl.gz", "wt", encoding="utf-8") as f:
        for issue in issues[-limit:]:
            f.write(json.dumps(issue) + "\n")
    (out_dir / "meta.json").write_text(json.dumps({"repo": repo, "issues": min(limit, len(issues))}), encoding="utf-8")
