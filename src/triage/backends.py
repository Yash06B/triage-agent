"""Where the MCP server's tools read from and write to: a frozen snapshot, or a live GitHub repo."""

from __future__ import annotations

from typing import Protocol

import httpx

from triage.store import Issue, IssueStore, Label


class Backend(Protocol):
    repo: str

    def list_labels(self) -> list[Label]: ...
    def search_issues(self, query: str, limit: int, state: str) -> list[Issue]: ...
    def get_issue(self, number: int) -> Issue | None: ...
    def add_labels(self, number: int, labels: list[str]) -> None: ...
    def post_comment(self, number: int, body: str) -> None: ...


class SnapshotBackend:
    """Read-only: the benchmark never writes anywhere, so writes are refused outright."""

    def __init__(self, repo: str, store: IssueStore):
        self.repo = repo
        self.store = store

    def list_labels(self) -> list[Label]:
        return self.store.labels

    def search_issues(self, query: str, limit: int, state: str) -> list[Issue]:
        return self.store.search(query, limit, state)

    def get_issue(self, number: int) -> Issue | None:
        return self.store.get(number)

    def add_labels(self, number: int, labels: list[str]) -> None:
        raise PermissionError("snapshot backend is read-only")

    def post_comment(self, number: int, body: str) -> None:
        raise PermissionError("snapshot backend is read-only")


def _issue_from_rest(d: dict) -> Issue:
    return Issue(
        number=d["number"],
        title=d["title"],
        body=(d.get("body") or "")[:4000],
        state=d["state"],
        labels=tuple(label["name"] for label in d.get("labels", [])),
        created_at=d["created_at"],
        closed_at=d.get("closed_at"),
        state_reason=d.get("state_reason"),
    )


class GitHubBackend:
    def __init__(self, repo: str, token: str, *, hide: set[int] | frozenset[int] = frozenset()):
        self.repo = repo
        self.hide = hide
        self._client = httpx.Client(
            base_url="https://api.github.com",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30,
        )
        self._labels: list[Label] | None = None

    def list_labels(self) -> list[Label]:
        if self._labels is None:
            labels, page = [], 1
            while True:
                r = self._client.get(f"/repos/{self.repo}/labels", params={"per_page": 100, "page": page})
                r.raise_for_status()
                batch = r.json()
                labels += [Label(d["name"], d.get("description") or "") for d in batch]
                if len(batch) < 100:
                    break
                page += 1
            self._labels = labels
        return self._labels

    def search_issues(self, query: str, limit: int, state: str) -> list[Issue]:
        q = f"{query} repo:{self.repo} is:issue"
        if state != "all":
            q += f" state:{state}"
        r = self._client.get("/search/issues", params={"q": q, "per_page": limit + len(self.hide)})
        r.raise_for_status()
        hits = [_issue_from_rest(d) for d in r.json()["items"] if d["number"] not in self.hide]
        return hits[:limit]

    def get_issue(self, number: int) -> Issue | None:
        if number in self.hide:
            return None
        r = self._client.get(f"/repos/{self.repo}/issues/{number}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        d = r.json()
        return None if "pull_request" in d else _issue_from_rest(d)

    def get_issue_unscoped(self, number: int) -> Issue:
        """The issue being triaged. `get_issue` hides it from the agent's own lookups."""
        r = self._client.get(f"/repos/{self.repo}/issues/{number}")
        r.raise_for_status()
        return _issue_from_rest(r.json())

    def add_labels(self, number: int, labels: list[str]) -> None:
        r = self._client.post(f"/repos/{self.repo}/issues/{number}/labels", json={"labels": labels})
        r.raise_for_status()

    def post_comment(self, number: int, body: str) -> None:
        r = self._client.post(f"/repos/{self.repo}/issues/{number}/comments", json={"body": body})
        r.raise_for_status()
