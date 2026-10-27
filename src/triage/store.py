"""An issue snapshot with BM25 search, scoped to what was visible at one point in time.

When the benchmark triages issue #N, the store is opened "as of" #N's creation time: issues filed
later don't exist, issues closed later still show as open, and #N itself is hidden. Without this
the agent could find the answer by searching the future.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from rank_bm25 import BM25Okapi

_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i if in is it its me my no not of on "
    "or so that the this to was we when where which while with you your uv".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class Label:
    name: str
    description: str = ""


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    body: str
    state: str
    labels: tuple[str, ...]
    created_at: str
    closed_at: str | None = None
    state_reason: str | None = None
    duplicate_of: int | None = None  # ground truth only; never served to the agent

    @classmethod
    def from_dict(cls, d: dict) -> Issue:
        return cls(
            number=d["number"],
            title=d["title"],
            body=d.get("body") or "",
            state=d.get("state", "open"),
            labels=tuple(d.get("labels", ())),
            created_at=d["created_at"],
            closed_at=d.get("closed_at"),
            state_reason=d.get("state_reason"),
            duplicate_of=d.get("duplicate_of"),
        )


def load_snapshot(directory: str | Path) -> tuple[list[Issue], list[Label]]:
    directory = Path(directory)
    with gzip.open(directory / "issues.jsonl.gz", "rt", encoding="utf-8") as f:
        issues = [Issue.from_dict(json.loads(line)) for line in f if line.strip()]
    labels = [Label(**d) for d in json.loads((directory / "labels.json").read_text(encoding="utf-8"))]
    return issues, labels


def _as_of(issue: Issue, cutoff: str) -> Issue:
    """The issue as it looked at `cutoff`: if it was closed afterwards, it was still open then."""
    if issue.closed_at and issue.closed_at > cutoff:
        return replace(issue, state="open", state_reason=None, closed_at=None)
    return issue


class IssueStore:
    def __init__(
        self,
        issues: list[Issue],
        labels: list[Label],
        *,
        as_of: str | None = None,
        hide: set[int] | frozenset[int] = frozenset(),
    ):
        visible = [i for i in issues if i.number not in hide and (as_of is None or i.created_at < as_of)]
        if as_of is not None:
            visible = [_as_of(i, as_of) for i in visible]
        self.labels = labels
        self._issues = visible
        self._by_number = {i.number: i for i in visible}
        # Titles carry most of the signal in issue text, so they count twice.
        corpus = [tokenize(f"{i.title} {i.title} {i.body[:1500]}") for i in visible]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def __len__(self) -> int:
        return len(self._issues)

    def get(self, number: int) -> Issue | None:
        return self._by_number.get(number)

    def search(self, query: str, limit: int = 5, state: str = "all") -> list[Issue]:
        tokens = tokenize(query)
        if not tokens or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for idx in ranked:
            if scores[idx] <= 0:
                break
            issue = self._issues[idx]
            if state != "all" and issue.state != state:
                continue
            hits.append(issue)
            if len(hits) == limit:
                break
        return hits
