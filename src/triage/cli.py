"""Command line: fetch a snapshot, build and run the benchmark, or triage a live issue."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import anyio

from triage.agent import STRATEGIES, Taxonomy, plan_actions, triage
from triage.llm import DEFAULT_MODELS, make_model

DEFAULT_TYPES = "bug,enhancement,question,documentation"


def _taxonomy(labels, args) -> Taxonomy:
    return Taxonomy.from_labels(labels, args.types.split(","), args.area_prefix)


def _add_taxonomy_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--types", default=DEFAULT_TYPES, help="comma-separated type labels (exactly one is chosen)")
    p.add_argument("--area-prefix", default="area:", help="labels with this prefix are offered as areas")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default="ollama")
    p.add_argument("--model", help="model name (default: %s)" % ", ".join(f"{k}: {v}" for k, v in DEFAULT_MODELS.items()))


def cmd_fetch(args) -> None:
    from triage.fetch import fetch_snapshot

    out = Path(args.out) / args.repo.replace("/", "__")
    fetch_snapshot(args.repo, out, args.limit)
    print(f"Wrote {out}")


def cmd_build_eval(args) -> None:
    from triage.eval import build_eval_set, save_eval_set
    from triage.store import load_snapshot

    issues, labels = load_snapshot(args.snapshot)
    items = build_eval_set(
        issues,
        _taxonomy(labels, args),
        since=args.since,
        n_duplicates=args.duplicates,
        n_typed=args.typed,
        seed=args.seed,
    )
    save_eval_set(items, Path(args.out))
    dups = sum(1 for x in items if x.gold_duplicate_of)
    print(f"Wrote {len(items)} items ({dups} duplicates) to {args.out}")


def cmd_eval(args) -> None:
    from triage.eval import load_eval_set, run_eval
    from triage.store import load_snapshot

    items = load_eval_set(Path(args.dataset))[: args.limit]
    _, labels = load_snapshot(args.snapshot)
    model = make_model(args.provider, args.model)
    strategies = args.strategy or list(STRATEGIES)
    anyio.run(
        lambda: run_eval(
            Path(args.snapshot), items, model, strategies, _taxonomy(labels, args), Path(args.out), budget=args.budget
        )
    )


def cmd_report(args) -> None:
    from triage.eval import report, retrieval_ceiling, load_eval_set

    table = report(Path(args.results))
    print(table)
    if args.snapshot and args.dataset:
        ceiling = retrieval_ceiling(Path(args.snapshot), load_eval_set(Path(args.dataset)))
        print(f"\nTitle search finds the true duplicate in the top {ceiling['k']} for "
              f"{ceiling['found']}/{ceiling['duplicates']} duplicates ({100 * (ceiling['recall'] or 0):.0f}%).")


def cmd_run(args) -> None:
    """Triage one live issue: the agent investigates via the MCP server, you approve each write."""
    from mcp import StdioServerParameters
    from mcp.client.client import Client

    from triage.backends import GitHubBackend
    from triage.fetch import github_token

    github = GitHubBackend(args.repo, github_token())
    issue = github.get_issue_unscoped(args.number)
    model = make_model(args.provider, args.model)
    writes = "live" if args.apply else "dry-run"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "triage.mcp_server", "--repo", args.repo, "--hide", str(args.number), "--writes", writes],
    )

    async def main() -> None:
        async with Client(server) as client:
            taxonomy = _taxonomy(github.list_labels(), args)
            print(f"Triaging {args.repo}#{issue.number}: {issue.title}\n  with {model.name} ({args.strategy})\n")
            result = await triage(
                client, model, issue, taxonomy, repo=args.repo, strategy=args.strategy, budget=args.budget
            )
            for call in result.tool_calls:
                print(f"  -> {call['name']}({json.dumps(call['arguments'])})")
            if result.decision is None:
                sys.exit(f"No decision: {result.error}")
            d = result.decision
            print(f"\nType: {d.type}   Areas: {', '.join(d.areas) or 'none'}   Duplicate of: {d.duplicate_of or 'none'}")
            print(f"Reasoning: {d.reasoning}\n")
            _save_trace(result, args.repo)

            for tool, arguments in plan_actions(issue.number, d):
                preview = arguments.get("labels") or arguments.get("body")
                if not args.yes:
                    answer = input(f"{tool} on #{issue.number}: {preview}\n  Run it{' (dry run)' if writes == 'dry-run' else ''}? [y/N] ")
                    if answer.strip().lower() != "y":
                        print("  skipped")
                        continue
                from triage.agent import call_mcp

                print("  " + await call_mcp(client, tool, arguments))

    anyio.run(main)


def _save_trace(result, repo: str) -> None:
    runs = Path("runs")
    runs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = runs / f"{repo.replace('/', '__')}-{result.issue}-{stamp}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"Trace saved to {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="triage", description="GitHub issue triage agent over MCP.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="snapshot a repository's issues for offline evaluation")
    p.add_argument("repo", help="owner/name")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--out", default="data")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build-eval", help="sample a labeled eval set from a snapshot")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--since", default="2025-07-01", help="only issues filed after this date (they have history to search)")
    p.add_argument("--duplicates", type=int, default=40)
    p.add_argument("--typed", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    _add_taxonomy_args(p)
    p.set_defaults(func=cmd_build_eval)

    p = sub.add_parser("eval", help="run strategies over an eval set (resumable)")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default="eval/results")
    p.add_argument("--strategy", action="append", choices=STRATEGIES, help="repeatable; default: all")
    p.add_argument("--limit", type=int)
    p.add_argument("--budget", type=int, default=4, help="max tool calls per issue for the agent")
    _add_model_args(p)
    _add_taxonomy_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("report", help="print the results table")
    p.add_argument("--results", default="eval/results")
    p.add_argument("--snapshot", help="with --dataset, also report the retrieval ceiling")
    p.add_argument("--dataset")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("run", help="triage one live issue, approving each write")
    p.add_argument("repo", help="owner/name")
    p.add_argument("number", type=int)
    p.add_argument("--strategy", choices=STRATEGIES, default="agent")
    p.add_argument("--budget", type=int, default=4)
    p.add_argument("--apply", action="store_true", help="write to GitHub for real (default: dry run)")
    p.add_argument("--yes", action="store_true", help="approve every action without asking")
    _add_model_args(p)
    _add_taxonomy_args(p)
    p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    for noisy in ("httpx", "mcp"):  # per-request INFO lines drown out the progress output
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
