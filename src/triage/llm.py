"""Tool-calling chat for Ollama (local, free) and Claude, behind one small interface.

A `Conversation` holds the provider's own message format, so the agent loop never has to know
which one it is talking to. `step(force=...)` makes the model call a specific tool: Claude has
`tool_choice` for that; Ollama doesn't, so the forced call is made with a JSON-schema constrained
response instead, which works even on models with shaky native tool calling.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

DEFAULT_MODELS = {"ollama": "llama3.2:3b", "anthropic": "claude-haiku-5-5"}


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class Conversation(Protocol):
    def step(self, force: str | None = None) -> Turn: ...
    def add_tool_results(self, results: list[tuple[ToolCall, str]]) -> None: ...
    def add_user(self, text: str) -> None: ...


class ChatModel(Protocol):
    name: str

    def conversation(self, system: str, user: str, tools: list[ToolSpec]) -> Conversation: ...


class OllamaModel:
    def __init__(self, model: str, base_url: str = "http://localhost:11434", num_ctx: int = 8192, max_tokens: int = 1024):
        self.name = f"ollama/{model}"
        self.model = model
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens  # small models occasionally loop; cap them instead of waiting minutes
        self._client = httpx.Client(base_url=base_url, timeout=600)

    def conversation(self, system: str, user: str, tools: list[ToolSpec]) -> OllamaConversation:
        return OllamaConversation(self, system, user, tools)

    def chat(self, body: dict) -> dict:
        body = {
            "model": self.model,
            "stream": False,
            "think": False,  # thinking models (qwen3) answer far faster without it; ignored by the rest
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": self.max_tokens},
            **body,
        }
        r = self._client.post("/api/chat", json=body)
        r.raise_for_status()
        return r.json()


class OllamaConversation:
    def __init__(self, model: OllamaModel, system: str, user: str, tools: list[ToolSpec]):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self._calls = 0

    def _call_id(self) -> str:
        self._calls += 1
        return f"call_{self._calls}"

    def step(self, force: str | None = None) -> Turn:
        if force:
            return self._forced(self.tools[force])
        tools = [{"type": "function", "function": vars(t)} for t in self.tools.values()]
        data = self.model.chat({"messages": self.messages, "tools": tools})
        message = data["message"]
        self.messages.append(message)
        calls = [
            ToolCall(self._call_id(), c["function"]["name"], _as_dict(c["function"].get("arguments")))
            for c in message.get("tool_calls") or []
        ]
        return Turn(message.get("content") or "", calls, data.get("prompt_eval_count", 0), data.get("eval_count", 0))

    def _forced(self, tool: ToolSpec) -> Turn:
        self.messages.append(
            {"role": "user", "content": f"Now call {tool.name}. Reply with only its JSON arguments."}
        )
        data = self.model.chat({"messages": self.messages, "format": tool.parameters})
        content = data["message"].get("content") or ""
        self.messages.append({"role": "assistant", "content": content})
        call = ToolCall(self._call_id(), tool.name, _as_dict(content))
        return Turn("", [call], data.get("prompt_eval_count", 0), data.get("eval_count", 0))

    def add_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:
        for call, output in results:
            self.messages.append({"role": "tool", "tool_name": call.name, "content": output})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


class AnthropicModel:
    def __init__(self, model: str, max_tokens: int = 1024):
        import anthropic

        self.name = f"anthropic/{model}"
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def conversation(self, system: str, user: str, tools: list[ToolSpec]) -> AnthropicConversation:
        return AnthropicConversation(self, system, user, tools)


class AnthropicConversation:
    def __init__(self, model: AnthropicModel, system: str, user: str, tools: list[ToolSpec]):
        self.model = model
        self.system = system
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": user}]

    def step(self, force: str | None = None) -> Turn:
        kwargs: dict[str, Any] = {}
        if force:
            kwargs["tool_choice"] = {"type": "tool", "name": force}
        msg = self.model._client.messages.create(
            model=self.model.model,
            max_tokens=self.model.max_tokens,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            **kwargs,
        )
        self.messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in msg.content]})
        text = "".join(b.text for b in msg.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in msg.content if b.type == "tool_use"]
        return Turn(text, calls, msg.usage.input_tokens, msg.usage.output_tokens)

    def add_tool_results(self, results: list[tuple[ToolCall, str]]) -> None:
        blocks = [{"type": "tool_result", "tool_use_id": call.id, "content": output} for call, output in results]
        self.messages.append({"role": "user", "content": blocks})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


def _as_dict(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def make_model(provider: str, model: str | None = None) -> ChatModel:
    model = model or DEFAULT_MODELS[provider]
    if provider == "ollama":
        return OllamaModel(model, os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    if provider == "anthropic":
        return AnthropicModel(model)
    raise ValueError(f"unknown provider {provider!r}")
