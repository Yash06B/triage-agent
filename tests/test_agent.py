from __future__ import annotations

import anyio
from mcp.client.client import Client

from triage.agent import plan_actions, triage
from triage.backends import SnapshotBackend
from triage.eval import unlabeled
from triage.llm import ToolCall, Turn
from triage.mcp_server import build_server
from triage.store import IssueStore


class ScriptedModel:
    """Replays a fixed list of turns and records what it was shown."""

    name = "fake/scripted"

    def __init__(self, turns: list[Turn]):
        self.turns = turns
        self.forced: list[str | None] = []
        self.tool_results: list[str] = []
        self.system = self.user = ""
        self.tools: list[str] = []

    def conversation(self, system, user, tools):
        self.system, self.user, self.tools = system, user, [t.name for t in tools]
        return self

    def step(self, force=None):
        self.forced.append(force)
        if force:
            # A forced turn always produces the forced call, like tool_choice / constrained JSON.
            for turn in self.turns:
                if turn.tool_calls and turn.tool_calls[0].name == force:
                    self.turns.remove(turn)
                    return turn
        return self.turns.pop(0)

    def add_tool_results(self, results):
        self.tool_results += [output for _, output in results]

    def add_user(self, text):
        pass


def submit(**args) -> Turn:
    base = {"reasoning": "r", "type": "bug", "areas": [], "duplicate_of": 0, "summary": "s"}
    return Turn("", [ToolCall("s", "submit_triage", base | args)], 10, 5)


def call(name: str, **args) -> Turn:
    return Turn("", [ToolCall(name, name, args)], 10, 5)


def _triage(issues, labels, taxonomy, target_number, model, strategy="agent", budget=4):
    target = next(i for i in issues if i.number == target_number)
    store = IssueStore(issues, labels, as_of=target.created_at, hide={target.number})

    async def main():
        async with Client(build_server(SnapshotBackend("astral-sh/uv", store))) as client:
            return await triage(
                client, model, unlabeled(target), taxonomy, repo="astral-sh/uv", strategy=strategy, budget=budget
            )

    return anyio.run(main)


def test_agent_searches_reads_and_submits_a_duplicate(issues, labels, taxonomy):
    model = ScriptedModel(
        [
            call("search_issues", query="windows long path torch"),
            call("get_issue", number=3),
            submit(type="bug", areas=["area:windows"], duplicate_of=3),
        ]
    )
    result = _triage(issues, labels, taxonomy, 5, model)
    assert result.error is None
    assert result.decision.type == "bug"
    assert result.decision.areas == ["area:windows"]
    assert result.decision.duplicate_of == 3
    assert [c["name"] for c in result.tool_calls] == ["search_issues", "get_issue"]
    assert "#3 (open)" in model.tool_results[0]  # closed only after #5 was filed
    assert model.tools == ["search_issues", "get_issue", "submit_triage"]
    assert not result.forced_submit and result.input_tokens == 30


def test_invalid_labels_and_unknown_duplicates_are_dropped(issues, labels, taxonomy):
    model = ScriptedModel([submit(type="crash", areas=["area:cache", "area:gpu"], duplicate_of=999)])
    result = _triage(issues, labels, taxonomy, 6, model, strategy="zero-shot")
    d = result.decision
    assert d.type is None and d.areas == ["area:cache"]
    assert d.invalid_labels == ["crash", "area:gpu"]
    assert d.duplicate_of is None and d.invalid_duplicate == 999


def test_issue_cannot_be_a_duplicate_of_itself_or_of_a_later_issue(issues, labels, taxonomy):
    model = ScriptedModel([submit(duplicate_of=6)])  # #6 was filed after #5
    assert _triage(issues, labels, taxonomy, 5, model, strategy="zero-shot").decision.invalid_duplicate == 6


def test_agent_is_forced_to_submit_when_the_budget_runs_out(issues, labels, taxonomy):
    model = ScriptedModel([call("search_issues", query="cache"), call("search_issues", query="prune"), submit()])
    result = _triage(issues, labels, taxonomy, 6, model, budget=2)
    assert result.forced_submit
    assert model.forced == [None, None, "submit_triage"]


def test_prose_replies_are_nudged(issues, labels, taxonomy):
    model = ScriptedModel([Turn("I think this is a bug."), call("search_issues", query="cache"), submit()])
    result = _triage(issues, labels, taxonomy, 6, model)
    assert result.nudges == 1 and result.decision.type == "bug"


def test_zero_shot_and_rag_submit_in_one_forced_turn(issues, labels, taxonomy):
    zero = ScriptedModel([submit()])
    result = _triage(issues, labels, taxonomy, 5, zero, strategy="zero-shot")
    assert zero.forced == ["submit_triage"] and zero.tools == ["submit_triage"] and result.tool_calls == []

    rag = ScriptedModel([submit(duplicate_of=3)])
    result = _triage(issues, labels, taxonomy, 5, rag, strategy="rag")
    assert "Similar existing issues:" in rag.user and "#3" in rag.user
    assert result.tool_calls[0]["by"] == "harness" and result.decision.duplicate_of == 3


def test_plan_actions_labels_and_comments(issues, labels, taxonomy):
    model = ScriptedModel([submit(type="bug", areas=["area:windows"], duplicate_of=3, summary="Long paths fail.")])
    decision = _triage(issues, labels, taxonomy, 5, model, strategy="zero-shot").decision
    actions = plan_actions(5, decision)
    assert actions[0] == ("add_labels", {"number": 5, "labels": ["bug", "area:windows"]})
    assert actions[1][0] == "post_comment" and "duplicate of #3" in actions[1][1]["body"]


def test_ticking_every_area_keeps_none(issues, labels, taxonomy):
    model = ScriptedModel([submit(areas=["area:windows", "area:cache"] * 2)])
    assert _triage(issues, labels, taxonomy, 6, model, strategy="zero-shot").decision.areas == ["area:cache", "area:windows"]

    many = [f"area:{n}" for n in ("windows", "cache")]
    taxonomy.areas.extend([type(taxonomy.areas[0])(f"area:x{i}") for i in range(3)])
    model = ScriptedModel([submit(areas=many + [f"area:x{i}" for i in range(3)])])
    d = _triage(issues, labels, taxonomy, 6, model, strategy="zero-shot").decision
    assert d.areas == [] and d.areas_rejected


def test_list_arguments_sent_as_strings_are_parsed(issues, labels, taxonomy):
    for areas in ("['area:cache']", '["area:cache"]', "area:cache", "area:cache, area:gpu"):
        model = ScriptedModel([submit(type="Bug", areas=areas)])
        d = _triage(issues, labels, taxonomy, 6, model, strategy="zero-shot").decision
        assert d.type == "bug" and d.areas == ["area:cache"], areas
        assert d.invalid_labels in ([], ["area:gpu"])


def test_agent_must_search_once_before_submitting(issues, labels, taxonomy):
    model = ScriptedModel([submit(), call("search_issues", query="cache prune"), submit()])
    result = _triage(issues, labels, taxonomy, 6, model)
    assert result.premature_submits == 1
    assert [c["name"] for c in result.tool_calls] == ["search_issues"]
    assert model.tool_results[0].startswith("Error: search existing issues")
