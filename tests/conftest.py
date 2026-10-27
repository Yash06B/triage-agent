from __future__ import annotations

import pytest

from triage.agent import Taxonomy
from triage.store import Issue, Label

LABELS = [
    Label("bug", "Something isn't working"),
    Label("enhancement", "New feature or improvement"),
    Label("question", "Asking for clarification or support"),
    Label("documentation", "Improvements or additions to documentation"),
    Label("area:windows", "Specific to Windows"),
    Label("area:cache", "Caching of packages and metadata"),
    Label("duplicate", "This issue already exists"),
]


def make_issue(number: int, title: str, body: str = "", **kw) -> Issue:
    kw.setdefault("created_at", f"2026-01-{number:02d}T00:00:00Z")
    kw.setdefault("state", "open")
    kw.setdefault("labels", ())
    return Issue(number=number, title=title, body=body, **kw)


@pytest.fixture
def issues() -> list[Issue]:
    return [
        make_issue(1, "Cache directory grows without bound", "uv cache keeps growing on every sync", labels=("bug", "area:cache")),
        make_issue(2, "Add a --frozen flag to uv tool install", "Please add frozen installs for tools", labels=("enhancement",)),
        make_issue(
            3,
            "Long path error on Windows when installing torch",
            "error: failed to create directory: path too long",
            labels=("bug", "area:windows"),
            state="closed",
            closed_at="2026-01-20T00:00:00Z",
            state_reason="completed",
        ),
        make_issue(4, "How do I pin a Python version per project?", "question about .python-version", labels=("question",)),
        make_issue(
            5,
            "Windows: path too long installing torch",
            "Same long path failure on Windows",
            labels=("duplicate",),
            state="closed",
            closed_at="2026-01-06T00:00:00Z",
            state_reason="duplicate",
            duplicate_of=3,
        ),
        make_issue(6, "Cache prune deletes too much", "uv cache prune removed wheels I still need", labels=("bug", "area:cache")),
    ]


@pytest.fixture
def labels() -> list[Label]:
    return LABELS


@pytest.fixture
def taxonomy() -> Taxonomy:
    return Taxonomy.from_labels(LABELS, ["bug", "enhancement", "question", "documentation"], "area:")
