import anyio
from mcp.client.client import Client

from triage.backends import SnapshotBackend
from triage.mcp_server import build_server
from triage.store import IssueStore


def _server(issues, labels, writes="off"):
    return build_server(SnapshotBackend("astral-sh/uv", IssueStore(issues, labels)), writes)


def _run(server, fn):
    async def main():
        async with Client(server) as client:
            return await fn(client)

    return anyio.run(main)


def test_read_only_server_exposes_no_write_tools(issues, labels):
    async def names(client):
        return {t.name for t in (await client.list_tools()).tools}

    assert _run(_server(issues, labels), names) == {"list_labels", "search_issues", "get_issue"}
    assert {"add_labels", "post_comment"} <= _run(_server(issues, labels, "dry-run"), names)


def test_search_returns_compact_lines(issues, labels):
    async def search(client):
        return await client.call_tool("search_issues", {"query": "cache prune", "limit": 2})

    result = _run(_server(issues, labels), search)
    text = result.content[0].text
    assert text.startswith("#6 (open) [bug, area:cache] Cache prune deletes too much")
    assert len(text.splitlines()) == 4  # two hits, each with a snippet line


def test_get_missing_issue_is_a_tool_error(issues, labels):
    async def get(client):
        return await client.call_tool("get_issue", {"number": 999})

    result = _run(_server(issues, labels), get)
    assert result.is_error and "#999 not found" in result.content[0].text


def test_dry_run_write_does_nothing_but_validates_labels(issues, labels):
    async def label(client):
        ok = await client.call_tool("add_labels", {"number": 1, "labels": ["bug"]})
        bad = await client.call_tool("add_labels", {"number": 1, "labels": ["area:made-up"]})
        return ok, bad

    ok, bad = _run(_server(issues, labels, "dry-run"), label)
    assert ok.content[0].text == "DRY RUN: would add bug to #1."
    assert bad.is_error and "area:made-up" in bad.content[0].text


def test_list_labels_filters_by_prefix(issues, labels):
    async def areas(client):
        return await client.call_tool("list_labels", {"prefix": "area:"})

    text = _run(_server(issues, labels), areas).content[0].text
    assert text.splitlines() == ["area:windows: Specific to Windows", "area:cache: Caching of packages and metadata"]
