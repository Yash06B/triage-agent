"""MCP server exposing issue-tracker tools: search, read, label, comment.

Any MCP client can use it (Claude Desktop, Claude Code, the agent in this repo). Write tools have
three modes: `off` (not registered at all), `dry-run` (they describe what they would do), and
`live` (they call GitHub). Results are compact text rather than JSON, because small local models
read prose more reliably than nested objects.

    triage-mcp --repo owner/name --writes dry-run         # live GitHub, reads only for real
    triage-mcp --snapshot data/astral-sh__uv --as-of 2026-09-01T00:00:00Z --hide 12345
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from triage.backends import Backend, GitHubBackend, SnapshotBackend
from triage.store import Issue, IssueStore, load_snapshot

WriteMode = Literal["off", "dry-run", "live"]
READ_ONLY = ToolAnnotations(read_only_hint=True)
ADDITIVE = ToolAnnotations(read_only_hint=False, destructive_hint=False)


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_hit(issue: Issue) -> str:
    state = issue.state if not issue.state_reason else f"{issue.state}: {issue.state_reason}"
    labels = f" [{', '.join(issue.labels)}]" if issue.labels else ""
    return f"#{issue.number} ({state}){labels} {issue.title}\n    {_one_line(issue.body, 160)}"


def format_issue(issue: Issue, body_chars: int = 2500) -> str:
    state = issue.state if not issue.state_reason else f"{issue.state} ({issue.state_reason})"
    body = issue.body if len(issue.body) <= body_chars else issue.body[:body_chars] + "\n[…truncated]"
    return (
        f"#{issue.number}: {issue.title}\n"
        f"State: {state} · Opened: {issue.created_at[:10]} · Labels: {', '.join(issue.labels) or 'none'}\n\n"
        f"{body or '(no description)'}"
    )


def build_server(backend: Backend, writes: WriteMode = "off") -> MCPServer:
    server = MCPServer(
        "issue-triage",
        instructions=f"Tools for triaging issues in the GitHub repository {backend.repo}.",
    )

    @server.tool(annotations=READ_ONLY)
    def list_labels(prefix: str = "") -> str:
        """List the repository's labels with their descriptions. Pass a prefix such as "area:" to filter."""
        labels = [lb for lb in backend.list_labels() if lb.name.startswith(prefix)]
        return "\n".join(f"{lb.name}: {lb.description}" if lb.description else lb.name for lb in labels) or "No labels."

    @server.tool(annotations=READ_ONLY)
    def search_issues(query: str, limit: int = 5, state: Literal["all", "open", "closed"] = "all") -> str:
        """Keyword search over existing issues (title and body). Use it to find duplicates or related reports.
        Short, specific queries work best: a command name, an error message, a flag.
        Leave state as "all": the issue a new report duplicates has often already been closed."""
        # Small models like to ask for a single hit, which is rarely enough to spot a duplicate.
        hits = backend.search_issues(query, max(3, min(limit, 10)), state)
        return "\n".join(format_hit(i) for i in hits) if hits else "No matching issues."

    @server.tool(annotations=READ_ONLY)
    def get_issue(number: int) -> str:
        """Read one issue in full: title, state, labels and description."""
        issue = backend.get_issue(number)
        if issue is None:
            raise ToolError(f"Issue #{number} not found.")
        return format_issue(issue)

    if writes == "off":
        return server

    @server.tool(annotations=ADDITIVE)
    def add_labels(number: int, labels: list[str]) -> str:
        """Add labels to an issue. Only existing repository labels are accepted."""
        known = {lb.name for lb in backend.list_labels()}
        unknown = [lb for lb in labels if lb not in known]
        if unknown:
            raise ToolError(f"Unknown labels: {', '.join(unknown)}. Call list_labels to see valid ones.")
        if writes == "dry-run":
            return f"DRY RUN: would add {', '.join(labels)} to #{number}."
        backend.add_labels(number, labels)
        return f"Added {', '.join(labels)} to #{number}."

    @server.tool(annotations=ADDITIVE)
    def post_comment(number: int, body: str) -> str:
        """Post a comment on an issue."""
        if writes == "dry-run":
            return f"DRY RUN: would comment on #{number}:\n{body}"
        backend.post_comment(number, body)
        return f"Commented on #{number}."

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="triage-mcp", description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="live GitHub repository, owner/name (token from GITHUB_TOKEN or gh)")
    source.add_argument("--snapshot", help="directory written by `triage fetch`")
    parser.add_argument("--as-of", help="snapshot only: show the tracker as it was at this ISO timestamp")
    parser.add_argument("--hide", type=int, action="append", default=[], help="issue number to hide (repeatable)")
    parser.add_argument("--writes", choices=["off", "dry-run", "live"], default="off")
    args = parser.parse_args(argv)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.snapshot:
        if args.writes == "live":
            parser.error("--writes live needs --repo; a snapshot is read-only")
        issues, labels = load_snapshot(args.snapshot)
        store = IssueStore(issues, labels, as_of=args.as_of, hide=set(args.hide))
        repo = _snapshot_repo(args.snapshot)
        backend: Backend = SnapshotBackend(repo, store)
    else:
        from triage.fetch import github_token

        backend = GitHubBackend(args.repo, github_token(), hide=set(args.hide))
    build_server(backend, args.writes).run("stdio")


def _snapshot_repo(directory: str) -> str:
    meta = Path(directory) / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8"))["repo"] if meta.exists() else Path(directory).name


if __name__ == "__main__":
    main()
