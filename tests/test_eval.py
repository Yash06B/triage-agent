from triage.eval import build_eval_set, score
from triage.fetch import _duplicate_of


def test_eval_set_keeps_only_duplicates_of_earlier_issues(issues, taxonomy):
    items = build_eval_set(issues, taxonomy, since="2026-01-01", n_duplicates=10, n_typed=10)
    by_number = {x.number: x for x in items}
    assert by_number[5].gold_duplicate_of == 3
    assert by_number[3].gold_type == "bug" and by_number[3].gold_areas == ["area:windows"]
    assert by_number[5].gold_type is None  # labeled "duplicate", not a type


def _record(gold_type, gold_dup, pred_type, pred_dup, areas=(), gold_areas=()):
    return {
        "gold": {"gold_type": gold_type, "gold_duplicate_of": gold_dup, "gold_areas": list(gold_areas)},
        "decision": {
            "type": pred_type,
            "areas": list(areas),
            "duplicate_of": pred_dup,
            "invalid_labels": [],
            "invalid_duplicate": None,
        },
        "tool_calls": [{"name": "search_issues"}, {"name": "search_issues", "by": "harness"}],
        "error": None,
        "forced_submit": False,
        "latency_s": 1.0,
        "input_tokens": 100,
        "output_tokens": 10,
    }


def test_score():
    m = score(
        [
            _record("bug", None, "bug", None, areas=["area:cache"], gold_areas=["area:cache"]),
            _record("question", None, "bug", 7),  # wrong type and a false duplicate
            _record(None, 3, "bug", 3),  # duplicate found
            _record(None, 4, "bug", None),  # duplicate missed
        ]
    )
    assert m["type_accuracy"] == 50.0
    assert m["dup_recall"] == 50.0
    assert m["dup_precision"] == 50.0
    assert m["false_dup_rate"] == 50.0
    assert m["area_precision"] == 100.0 and m["area_recall"] == 100.0
    assert m["avg_tool_calls"] == 1  # the harness's own search doesn't count


def test_duplicate_of_prefers_github_link_then_comment():
    node = {
        "stateReason": "DUPLICATE",
        "timelineItems": {"nodes": [{"duplicateOf": {"number": 42}}]},
        "comments": {"nodes": [{"body": "Duplicate of #7"}]},
    }
    assert _duplicate_of(node) == 42
    node["timelineItems"]["nodes"] = [{}]
    assert _duplicate_of(node) == 7
    node["stateReason"] = "COMPLETED"
    assert _duplicate_of(node) is None
