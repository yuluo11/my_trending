"""LLM client abstractions and adapters for analyst agents."""

from __future__ import annotations

import json
from typing import Any, Protocol


class LLMClient(Protocol):
    """Stable client interface consumed by analyst agents."""

    def invoke(self, prompt: str, *, payload: dict[str, Any] | None = None) -> Any:
        """Invoke the model for free-form output."""

    def invoke_json(
        self,
        prompt: str,
        *,
        payload: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke the model and normalize the output into a JSON object."""


class LLMRunnable(Protocol):
    """Minimal invocation interface for LangChain-compatible runnables."""

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        """Invoke the runnable with a prompt string or message list."""


class LangChainRunnableLLMClient:
    """Adapt a LangChain-style runnable/chat model into the shared client interface."""

    def __init__(self, runnable: LLMRunnable) -> None:
        self.runnable = runnable

    def invoke(self, prompt: str, *, payload: dict[str, Any] | None = None) -> Any:
        """Invoke the wrapped runnable with normalized prompt input."""
        return self.runnable.invoke(self._build_input(prompt, payload=payload))

    def invoke_json(
        self,
        prompt: str,
        *,
        payload: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke the runnable and parse the response as JSON when possible."""
        structured_prompt = prompt
        if schema:
            structured_prompt = (
                f"{prompt}\n\n"
                "Output schema:\n"
                f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
            )
        response = self.invoke(structured_prompt, payload=payload)
        return self._parse_json_response(response)

    def _build_input(self, prompt: str, *, payload: dict[str, Any] | None) -> Any:
        """Build a string or message-list input suitable for a runnable."""
        if payload is None:
            return prompt

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ModuleNotFoundError:
            return (
                f"{prompt}\n\n"
                "Context JSON:\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            )

        return [
            SystemMessage(content=prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
        ]

    def _parse_json_response(self, response: Any) -> dict[str, Any]:
        """Normalize runnable output into a JSON object."""
        if isinstance(response, dict):
            return response

        content = getattr(response, "content", response)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            content = "\n".join(part for part in parts if part)

        if not isinstance(content, str):
            return {"summary": str(content)}

        stripped = content.strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            fenced = self._extract_fenced_json(stripped)
            if fenced is None:
                return {"summary": stripped}
            try:
                parsed = json.loads(fenced)
            except json.JSONDecodeError:
                return {"summary": stripped}

        if isinstance(parsed, dict):
            return parsed
        return {"summary": stripped}

    def _extract_fenced_json(self, text: str) -> str | None:
        """Extract a JSON object from a fenced block when present."""
        marker = "```"
        if marker not in text:
            return None
        for part in text.split(marker):
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate
        return None


def ensure_llm_client(
    *,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> LLMClient | None:
    """Return a stable llm_client, adapting raw runnables when necessary."""
    if llm_client is not None:
        return llm_client
    if llm is not None:
        return LangChainRunnableLLMClient(llm)
    return None
