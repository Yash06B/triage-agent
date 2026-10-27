"""Benchmark: replay real, already-triaged issues and compare the agent's decisions to the maintainers'.

Ground truth is what the maintainers did: the type label they applied, any area labels, and, for
issues they closed as duplicates, which issue they pointed to. Each issue is triaged against the
tracker as it was when that issue was filed (see `IssueStore`), so later issues and later closures
can't leak the answer.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from mcp.client.client import Client

from triage.agent import Strategy, Taxonomy, triage
from triage.backends import SnapshotBackend
from triage.llm import ChatModel
from triage.mcp_server import build_server
from triage.store import Issue, IssueStore, Label, load_snapshot

EXCLUDE_LABEL_PREFIXES = ("bot:ai-policy",)  # spam closed by a bot, not triaged by a person


@dataclass(frozen=True)
class EvalItem:
    number: int
    title: str
    created_at: str
    gold_type: str | None
    gold_areas: list[str]
    gold_duplicate_of: int | None


def build_eval_set(
    issues: list[Issue],
    taxonomy: Taxonomy,
    *,
    since: str,
    n_duplicates: int,
    n_typed: int,
    seed: int = 0,
) -> list[EvalItem]:
    by_number = {i.number: i for i in issues}
    type_names = {lb.name for lb in taxonomy.types}
    area_names = {lb.name for lb in taxonomy.areas}

    def item(i: Issue, duplicate_of: int | None) -> EvalItem:
        types = type_names.intersection(i.labels)
        return EvalItem(
            number=i.number,
            title=i.title,
            created_at=i.created_at,
            gold_type=next(iter(types)) if len(types) == 1 else None,
            gold_areas=sorted(area_names.intersection(i.labels)),
            gold_duplicate_of=duplicate_of,
        )

    pool = [
        i
        for i in issues
        if i.created_at >= since and i.body.strip() and not any(lb.startswith(EXCLUDE_LABEL_PREFIXES) for lb in i.labels)
    ]
    duplicates = [
        i
        for i in pool
        if i.state_reason == "duplicate"
        and i.duplicate_of in by_number
        and by_number[i.duplicate_of].created_at < i.created_at
    ]
    typed = [i for i in pool if i.state_reason != "duplicate" and len(type_names.intersection(i.labels)) == 1]

    rng = random.Random(seed)
    chosen = [item(i, i.duplicate_of) for i in rng.sample(duplicates, min(n_duplicates, len(duplicates)))]
    chosen += [item(i, None) for i in rng.sample(typed, min(n_typed, len(typed)))]
    return sorted(chosen, key=lambda x: x.number)


def save_eval_set(items: list[EvalItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(vars(x)) + "\n" for x in items), encoding="utf-8")


def load_eval_set(path: Path) -> list[EvalItem]:
    return [EvalItem(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scoped_store(issues: list[Issue], labels: list[Label], target: Issue) -> IssueStore:
    return IssueStore(issues, labels, as_of=target.created_at, hide={target.number})


def unlabeled(issue: Issue) -> Issue:
    """The issue as it arrived: no labels, still open."""
    return Issue(issue.number, issue.title, issue.body, "open", (), issue.created_at)


async def run_eval(
    snapshot: Path,
    items: list[EvalItem],
    model: ChatModel,
    strategies: list[Strategy],
    taxonomy: Taxonomy,
    out_dir: Path,
    *,
    budget: int = 4,
    progress=lambda line: print(line, flush=True),
) -> None:
    """Run every strategy on every item, appending to one JSONL per (model, strategy). Re-running resumes."""
    issues, labels = load_snapshot(snapshot)
    by_number = {i.number: i for i in issues}
    repo = json.loads((snapshot / "meta.json").read_text(encoding="utf-8"))["repo"]
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {s: out_dir / f"{_slug(model.name)}__{s}.jsonl" for s in strategies}
    done = {s: _done(files[s]) for s in strategies}

    for n, item in enumerate(items, 1):
        todo = [s for s in strategies if item.number not in done[s]]
        if not todo:
            continue
        target = by_number[item.number]
        server = build_server(SnapshotBackend(repo, scoped_store(issues, labels, target)))
        async with Client(server) as client:
            for strategy in todo:
                result = await triage(
                    client, model, unlabeled(target), taxonomy, repo=repo, strategy=strategy, budget=budget
                )
                record = result.to_dict() | {"gold": vars(item)}
                with files[strategy].open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                d = result.decision
                progress(
                    f"[{n}/{len(items)}] #{item.number} {strategy:9} "
                    f"type={d.type if d else None} (gold {item.gold_type}) "
                    f"dup={d.duplicate_of if d else None} (gold {item.gold_duplicate_of}) "
                    f"tools={len(result.tool_calls)} {result.latency_s}s"
                    + (f" ERROR {result.error}" if result.error else "")
                )


def retrieval_ceiling(snapshot: Path, items: list[EvalItem], k: int = 5) -> dict:
    """How often a single title search surfaces the true duplicate: the most `rag` can hope for."""
    issues, labels = load_snapshot(snapshot)
    by_number = {i.number: i for i in issues}
    dups = [x for x in items if x.gold_duplicate_of]
    hits = 0
    for x in dups:
        store = scoped_store(issues, labels, by_number[x.number])
        if x.gold_duplicate_of in {i.number for i in store.search(x.title, k)}:
            hits += 1
    return {"k": k, "duplicates": len(dups), "found": hits, "recall": hits / len(dups) if dups else None}


def _slug(name: str) -> str:
    return name.replace("/", "_").replace(":", "-")


def _done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["issue"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


# ---------------------------------------------------------------------------------------- metrics


def score(records: list[dict]) -> dict:
    """Aggregate metrics for one (model, strategy) run."""

    def pct(num: int, den: int) -> float | None:
        return round(100 * num / den, 1) if den else None

    decided = [r for r in records if r["decision"]]
    typed = [r for r in decided if r["gold"]["gold_type"]]
    type_correct = sum(r["decision"]["type"] == r["gold"]["gold_type"] for r in typed)

    dups = [r for r in records if r["gold"]["gold_duplicate_of"]]
    non_dups = [r for r in records if not r["gold"]["gold_duplicate_of"]]

    def predicted_dup(r: dict) -> int | None:
        return r["decision"]["duplicate_of"] if r["decision"] else None

    dup_correct = sum(predicted_dup(r) == r["gold"]["gold_duplicate_of"] for r in dups)
    dup_predictions = sum(predicted_dup(r) is not None for r in records)
    false_dups = sum(predicted_dup(r) is not None for r in non_dups)

    tp = fp = fn = 0
    for r in decided:
        if r["gold"]["gold_duplicate_of"]:
            continue  # maintainers rarely label issues they close as duplicates
        pred, gold = set(r["decision"]["areas"]), set(r["gold"]["gold_areas"])
        tp, fp, fn = tp + len(pred & gold), fp + len(pred - gold), fn + len(gold - pred)

    model_calls = [sum(c.get("by") != "harness" for c in r["tool_calls"]) for r in records]
    return {
        "n": len(records),
        "type_accuracy": pct(type_correct, len(typed)),
        "dup_recall": pct(dup_correct, len(dups)),
        "dup_precision": pct(dup_correct, dup_predictions),
        "false_dup_rate": pct(false_dups, len(non_dups)),
        "area_precision": pct(tp, tp + fp),
        "area_recall": pct(tp, tp + fn),
        "invalid_labels": sum(len(r["decision"]["invalid_labels"]) for r in decided),
        "invalid_duplicates": sum(r["decision"]["invalid_duplicate"] is not None for r in decided),
        "areas_rejected_rate": pct(sum(r["decision"].get("areas_rejected", False) for r in decided), len(decided)),
        "errors": sum(r["error"] is not None or r["decision"] is None for r in records),
        "forced_submit_rate": pct(sum(r["forced_submit"] for r in records), len(records)),
        "avg_tool_calls": round(statistics.mean(model_calls), 2) if records else None,
        "p50_latency_s": round(statistics.median(r["latency_s"] for r in records), 1) if records else None,
        "avg_input_tokens": round(statistics.mean(r["input_tokens"] for r in records)) if records else None,
        "avg_output_tokens": round(statistics.mean(r["output_tokens"] for r in records)) if records else None,
        "n_typed": len(typed),
        "n_duplicates": len(dups),
    }


def load_results(results_dir: Path) -> dict[tuple[str, str], list[dict]]:
    runs: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(results_dir.glob("*__*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if records:
            runs[(records[0]["model"], records[0]["strategy"])] = records
    return runs


def report(results_dir: Path) -> str:
    rows = []
    header = (
        "| Model | Strategy | n | Type acc. | Dup recall | Dup precision | False-dup rate | Area P / R "
        "| Areas rejected | Tool calls | p50 latency | Tokens in / out | Forced submit |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    def fmt(v, suffix="%"):
        return "n/a" if v is None else f"{v}{suffix}"

    order = {s: i for i, s in enumerate(("zero-shot", "rag", "agent"))}
    for (model, strategy), records in sorted(load_results(results_dir).items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9))):
        m = score(records)
        rows.append(
            f"| {model} | {strategy} | {m['n']} | {fmt(m['type_accuracy'])} | {fmt(m['dup_recall'])} "
            f"| {fmt(m['dup_precision'])} | {fmt(m['false_dup_rate'])} "
            f"| {fmt(m['area_precision'])} / {fmt(m['area_recall'])} | {fmt(m['areas_rejected_rate'])} | {m['avg_tool_calls']} "
            f"| {m['p50_latency_s']} s | {m['avg_input_tokens']:,} / {m['avg_output_tokens']:,} "
            f"| {fmt(m['forced_submit_rate'])} |"
        )
    return "\n".join([header, *rows])
