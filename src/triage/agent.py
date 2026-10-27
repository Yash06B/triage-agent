"""The triage loop: a model reads a new issue, investigates through MCP tools, and submits a decision.

Three strategies share one code path so the benchmark compares like with like:

- `zero-shot`: the model sees only the issue and the label set, then must submit.
- `rag`: the harness runs one search on the issue title and puts the top hits in the prompt.
- `agent`: the model gets `search_issues` and `get_issue` and decides what to look at itself.

The model never writes to GitHub. It only *proposes* a decision through `submit_triage`; turning
that into labels and comments is `plan_actions`, and a human approves each one (see `cli.py`).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import anyio
from mcp.client.client import Client

from triage.llm import ChatModel, ToolCall, ToolSpec
from triage.store import Issue, Label

Strategy = Literal["zero-shot", "rag", "agent"]
STRATEGIES: tuple[Strategy, ...] = ("zero-shot", "rag", "agent")
AGENT_TOOLS = ("search_issues", "get_issue")
MAX_TOOL_OUTPUT = 3000
MAX_AREAS = 3  # more than this is a model ticking every box, not a judgement
ISSUE_BODY_CHARS = 3000


@dataclass(frozen=True)
class Taxonomy:
    types: list[Label]
    areas: list[Label]

    @classmethod
    def from_labels(cls, labels: list[Label], type_names: list[str], area_prefix: str = "area:") -> Taxonomy:
        by_name = {lb.name: lb for lb in labels}
        types = [by_name.get(name, Label(name)) for name in type_names]
        areas = [lb for lb in labels if area_prefix and lb.name.startswith(area_prefix)]
        return cls(types, areas)

    def describe(self) -> str:
        lines = ["Type labels (choose exactly one):"]
        lines += [f"- {lb.name}: {lb.description}" for lb in self.types]
        if self.areas:
            lines += ["", "Area labels (choose zero or more; only when clearly applicable):"]
            lines += [f"- {lb.name}: {lb.description}" if lb.description else f"- {lb.name}" for lb in self.areas]
        return "\n".join(lines)


def submit_tool(taxonomy: Taxonomy) -> ToolSpec:
    area_items: dict = {"type": "string"}
    if taxonomy.areas:
        area_items["enum"] = [lb.name for lb in taxonomy.areas]
    return ToolSpec(
        name="submit_triage",
        description="Submit your final triage decision for the issue. Call this exactly once, at the end.",
        parameters={
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "One or two sentences: what the issue is and why."},
                "type": {"type": "string", "enum": [lb.name for lb in taxonomy.types]},
                "areas": {"type": "array", "items": area_items},
                "duplicate_of": {
                    "type": "integer",
                    "description": "Number of an existing issue reporting the same problem, or 0 if none.",
                },
                "summary": {"type": "string", "description": "One-sentence summary for maintainers."},
            },
            "required": ["reasoning", "type", "areas", "duplicate_of", "summary"],
        },
    )


@dataclass
class Decision:
    type: str | None
    areas: list[str]
    duplicate_of: int | None
    summary: str = ""
    reasoning: str = ""
    invalid_labels: list[str] = field(default_factory=list)  # proposed labels that don't exist
    invalid_duplicate: int | None = None  # proposed duplicate that doesn't exist (or is the issue itself)
    areas_rejected: bool = False  # proposed more than MAX_AREAS areas, so none were kept
    submitted: dict = field(default_factory=dict)  # the raw submit_triage arguments, for the trace


@dataclass
class TriageResult:
    issue: int
    strategy: str
    model: str
    decision: Decision | None
    steps: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    nudges: int = 0  # turns where the model answered in prose instead of calling a tool
    premature_submits: int = 0  # agent tried to submit before searching even once
    forced_submit: bool = False  # ran out of steps and had to be made to submit
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM = """You triage newly opened GitHub issues for the {repo} repository.

For the issue you are given, decide:
1. type: exactly one type label.
2. areas: area labels that clearly apply, or none. Most issues have none.
3. duplicate_of: the number of an existing issue that reports the same underlying problem or asks for the same thing, or 0.
   Being on a related topic is not enough. If no existing issue clearly matches, use 0.

{taxonomy}
{process}"""

AGENT_PROCESS = """
How to work:
- Search existing issues with one to three short, specific queries (an error message, a command, a flag).
- If a result looks like the same problem, read it with get_issue to confirm before calling it a duplicate.
- Then call submit_triage. Use at most {budget} tool calls before submitting."""

RAG_PROCESS = """
The most similar existing issues, found by keyword search, are listed after the new issue. Use them only to judge duplicates.
Call submit_triage with your decision."""

ZERO_SHOT_PROCESS = """
Call submit_triage with your decision. You cannot search existing issues, so leave duplicate_of as 0 unless the issue itself names one."""


def issue_prompt(issue: Issue) -> str:
    body = issue.body if len(issue.body) <= ISSUE_BODY_CHARS else issue.body[:ISSUE_BODY_CHARS] + "\n[…truncated]"
    return f"New issue #{issue.number}: {issue.title}\n\n{body or '(no description)'}"


async def call_mcp(client: Client, name: str, arguments: dict) -> str:
    try:
        result = await client.call_tool(name, arguments)
    except Exception as exc:  # a malformed call from the model shouldn't end the run
        return f"Error: {exc}"
    text = "\n".join(getattr(block, "text", "") for block in result.content)
    if len(text) > MAX_TOOL_OUTPUT:
        text = text[:MAX_TOOL_OUTPUT] + "\n[…truncated]"
    return f"Error: {text}" if result.is_error else text


async def mcp_tool_specs(client: Client, allowed: tuple[str, ...]) -> list[ToolSpec]:
    listed = await client.list_tools()
    return [ToolSpec(t.name, t.description or "", t.input_schema) for t in listed.tools if t.name in allowed]


async def triage(
    client: Client,
    model: ChatModel,
    issue: Issue,
    taxonomy: Taxonomy,
    *,
    repo: str,
    strategy: Strategy = "agent",
    budget: int = 4,
) -> TriageResult:
    """Triage one issue. `client` is an MCP session whose search cannot see `issue` itself."""
    started = time.perf_counter()
    result = TriageResult(issue.number, strategy, model.name, decision=None)
    submit = submit_tool(taxonomy)
    user = issue_prompt(issue)

    if strategy == "agent":
        process = AGENT_PROCESS.format(budget=budget)
        tools = await mcp_tool_specs(client, AGENT_TOOLS) + [submit]
    elif strategy == "rag":
        process = RAG_PROCESS
        tools = [submit]
        hits = await call_mcp(client, "search_issues", {"query": issue.title, "limit": 5})
        result.tool_calls.append({"name": "search_issues", "arguments": {"query": issue.title}, "by": "harness"})
        user += f"\n\n---\nSimilar existing issues:\n{hits}"
    else:
        process = ZERO_SHOT_PROCESS
        tools = [submit]

    system = SYSTEM.format(repo=repo, taxonomy=taxonomy.describe(), process=process)
    convo = model.conversation(system, user, tools)
    used = 0
    max_turns = budget + 3  # room for a couple of prose replies before forcing a submit

    try:
        while result.decision is None:
            force = None
            if strategy != "agent" or used >= budget or result.steps >= max_turns - 1:
                force = submit.name
                result.forced_submit = strategy == "agent"
            turn = await anyio.to_thread.run_sync(convo.step, force)
            result.steps += 1
            result.input_tokens += turn.input_tokens
            result.output_tokens += turn.output_tokens

            if not turn.tool_calls:
                result.nudges += 1
                convo.add_user("Use the tools to investigate, or call submit_triage if you are done.")
                continue

            outputs: list[tuple[ToolCall, str]] = []
            for call in turn.tool_calls:
                t0 = time.perf_counter()
                if call.name == submit.name:
                    if strategy == "agent" and used == 0 and not force and not result.premature_submits:
                        # Duplicate detection is the point of having tools; one search is required.
                        result.premature_submits += 1
                        outputs.append((call, "Error: search existing issues at least once before submitting."))
                        continue
                    result.decision = await _validate(client, call.arguments, issue, taxonomy)
                    break
                if call.name in AGENT_TOOLS and strategy == "agent":
                    output = await call_mcp(client, call.name, call.arguments)
                    used += 1
                else:
                    output = f"Error: unknown tool {call.name}"
                result.tool_calls.append(
                    {"name": call.name, "arguments": call.arguments, "ms": round((time.perf_counter() - t0) * 1000)}
                )
                outputs.append((call, output))
            if result.decision is None:
                convo.add_tool_results(outputs)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    result.latency_s = round(time.perf_counter() - started, 2)
    return result


async def _validate(client: Client, args: dict, issue: Issue, taxonomy: Taxonomy) -> Decision:
    """Keep only what exists: real labels, and a duplicate that the tracker can actually find."""
    type_names = {lb.name for lb in taxonomy.types}
    area_names = {lb.name for lb in taxonomy.areas}
    raw_type = str(args.get("type") or "").strip()
    raw_type = next((t for t in type_names if t.lower() == raw_type.lower()), raw_type)
    areas = _as_list(args.get("areas"))
    valid_areas = sorted({a for a in areas if a in area_names})
    decision = Decision(
        type=raw_type if raw_type in type_names else None,
        areas=valid_areas if len(valid_areas) <= MAX_AREAS else [],
        duplicate_of=None,
        summary=str(args.get("summary") or ""),
        reasoning=str(args.get("reasoning") or ""),
        invalid_labels=[x for x in [raw_type, *areas] if x and x not in type_names | area_names],
        areas_rejected=len(valid_areas) > MAX_AREAS,
        submitted=args,
    )
    try:
        dup = int(args.get("duplicate_of") or 0)
    except (TypeError, ValueError):
        dup = 0
    if dup:
        exists = dup != issue.number and not (await call_mcp(client, "get_issue", {"number": dup})).startswith("Error")
        if exists:
            decision.duplicate_of = dup
        else:
            decision.invalid_duplicate = dup
    return decision


def _as_list(value) -> list[str]:
    """Small models sometimes send a list as a string: '["area:cli"]', "['area:cli']" or "area:cli, area:cache"."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value.replace("'", '"'))
        if isinstance(parsed, list):
            return _as_list(parsed)
    except json.JSONDecodeError:
        pass
    return [part.strip(" []\"'") for part in value.split(",") if part.strip(" []\"'")]


def plan_actions(number: int, decision: Decision) -> list[tuple[str, dict]]:
    """The writes a decision implies. Each one goes to a human for approval before it runs."""
    actions: list[tuple[str, dict]] = []
    labels = ([decision.type] if decision.type else []) + decision.areas
    if labels:
        actions.append(("add_labels", {"number": number, "labels": labels}))
    lines = [f"**Triage summary:** {decision.summary}"] if decision.summary else []
    if decision.duplicate_of:
        lines.append(f"This looks like a possible duplicate of #{decision.duplicate_of}.")
    if lines:
        lines.append("\n<sub>Drafted by an automated triage agent and approved by a maintainer.</sub>")
        actions.append(("post_comment", {"number": number, "body": "\n\n".join(lines)}))
    return actions
