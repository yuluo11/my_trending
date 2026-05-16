"""LangGraph-ready analyst agent primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypedDict

from ...knowledge.repository import DatasetName
from ...llm.client import LLMClient, LLMRunnable, ensure_llm_client
from ...tools.analyst.tooling import AnalystToolRegistry, ToolCallRequest, ToolCallResult
from .graph_analyst import KnowledgeBackedAnalystService


@dataclass(slots=True)
class AnalystTask:
    """Shared task payload passed into analyst agents."""

    subject: str
    symbol: str | None = None
    trade_date: str | None = None
    extra_context: str | None = None
    datasets: tuple[DatasetName, ...] | None = None
    metadata_filter: dict[str, Any] | None = None
    max_documents: int | None = None
    messages: list[Any] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: "AnalystRuntimeState") -> "AnalystTask":
        """Build a task object from LangGraph-style state."""
        return cls(
            subject=state["subject"],
            symbol=state.get("symbol"),
            trade_date=state.get("trade_date"),
            extra_context=state.get("extra_context"),
            datasets=tuple(state["datasets"]) if state.get("datasets") else None,
            metadata_filter=state.get("metadata_filter"),
            max_documents=state.get("max_documents"),
            messages=list(state.get("messages", [])),
        )


class AnalystRuntimeState(TypedDict, total=False):
    """Minimal state shared by analyst LangGraph nodes."""

    subject: str
    symbol: str | None
    trade_date: str | None
    extra_context: str | None
    datasets: tuple[DatasetName, ...] | list[DatasetName] | None
    metadata_filter: dict[str, Any] | None
    max_documents: int | None
    messages: list[Any]
    analyst_outputs: dict[str, dict[str, Any]]


class PromptProvider(Protocol):
    """Prompt source used by analyst agents."""

    def get_shared_prompt(self) -> str:
        """Return the shared prompt frame used by all analysts."""

    def get_analyst_prompt(self, analyst_name: str) -> str:
        """Return the analyst-specific prompt body."""


class StaticPromptProvider:
    """Small in-memory prompt provider for the first runtime iteration."""

    def __init__(
        self,
        *,
        shared_prompt: str,
        analyst_prompts: dict[str, str] | None = None,
    ) -> None:
        self.shared_prompt = shared_prompt.strip()
        self.analyst_prompts = dict(analyst_prompts or {})

    def get_shared_prompt(self) -> str:
        """Return the shared prompt frame."""
        return self.shared_prompt

    def get_analyst_prompt(self, analyst_name: str) -> str:
        """Return the analyst-specific prompt body when configured."""
        return self.analyst_prompts.get(analyst_name, "").strip()


class FilePromptProvider:
    """Load the shared frame and analyst-specific prompts from disk."""

    def __init__(
        self,
        prompts_dir: str | Path,
        *,
        shared_prompt_name: str = "base_prompt.txt",
        shared_dir_name: str = "shared",
        roles_dir_name: str = "roles",
        suffix: str = ".txt",
    ) -> None:
        self.prompts_dir = Path(prompts_dir)
        self.shared_prompt_name = shared_prompt_name
        self.shared_dir_name = shared_dir_name
        self.roles_dir_name = roles_dir_name
        self.suffix = suffix

    def get_shared_prompt(self) -> str:
        """Return the shared prompt frame from disk."""
        return self._read_prompt(self.prompts_dir / self.shared_dir_name / self.shared_prompt_name)

    def get_analyst_prompt(self, analyst_name: str) -> str:
        """Return the analyst-specific prompt body from disk."""
        return self._read_prompt(
            self.prompts_dir / self.roles_dir_name / f"{analyst_name}{self.suffix}"
        )

    def _read_prompt(self, prompt_path: Path) -> str:
        """Read a prompt file if it exists, otherwise return an empty prompt."""
        if not prompt_path.exists():
            return ""
        return prompt_path.read_text(encoding="utf-8").strip()


class BaseLangGraphAnalystAgent:
    """LangGraph-friendly analyst agent backed by tools and the knowledge base."""

    analyst_name = "knowledge_backed_analyst"

    def __init__(
        self,
        *,
        analyst_name: str | None = None,
        knowledge_service: KnowledgeBackedAnalystService,
        tool_registry: AnalystToolRegistry | None = None,
        prompt_provider: PromptProvider | None = None,
        llm_client: LLMClient | None = None,
        llm: LLMRunnable | None = None,
    ) -> None:
        if analyst_name:
            self.analyst_name = analyst_name
        self.knowledge_service = knowledge_service
        self.tool_registry = tool_registry or AnalystToolRegistry()
        self.prompt_provider = prompt_provider
        self.llm_client = ensure_llm_client(llm_client=llm_client, llm=llm)

    def plan_tool_calls(self, task: AnalystTask) -> list[ToolCallRequest]:
        """Plan the tool calls needed for the current analyst task."""
        if not self.tool_registry.has("search_knowledge_base"):
            return []
        return [
            ToolCallRequest(
                tool_name="search_knowledge_base",
                subject=task.subject,
                symbol=task.symbol,
                trade_date=task.trade_date,
                extra_context=task.extra_context,
                arguments={
                    "datasets": list(task.datasets) if task.datasets else None,
                    "metadata_filter": task.metadata_filter,
                    "max_documents": task.max_documents,
                },
            )
        ]

    def run_tools(self, tool_calls: list[ToolCallRequest]) -> list[ToolCallResult]:
        """Execute the planned tools in order."""
        return [self.tool_registry.invoke(tool_call) for tool_call in tool_calls]

    def retrieve_knowledge(
        self,
        task: AnalystTask,
        tool_results: list[ToolCallResult] | None = None,
    ) -> dict[str, Any]:
        """Retrieve the canonical knowledge payload for the task."""
        for result in tool_results or []:
            if result.success and result.tool_name == "search_knowledge_base":
                return dict(result.structured_data)

        return self.knowledge_service.analyze(
            task.subject,
            extra_context=task.extra_context,
            datasets=task.datasets,
            symbol=task.symbol,
            metadata_filter=task.metadata_filter,
            k=task.max_documents,
        )

    def build_prompt_context(
        self,
        task: AnalystTask,
        knowledge_payload: dict[str, Any],
        tool_results: list[ToolCallResult],
    ) -> dict[str, Any]:
        """Build the runtime context injected into the analyst prompt."""
        return {
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "extra_context": task.extra_context,
            "knowledge_payload": knowledge_payload,
            "tool_results": [self.serialize_tool_result(result) for result in tool_results],
        }

    def render_prompt(self, prompt_context: dict[str, Any]) -> str:
        """Render the shared and analyst-specific prompt blocks."""
        if self.prompt_provider is None:
            return ""

        shared_prompt = self.prompt_provider.get_shared_prompt()
        analyst_prompt = self.prompt_provider.get_analyst_prompt(self.analyst_name)
        context_lines = [
            f"subject: {prompt_context.get('subject', '')}",
            f"symbol: {prompt_context.get('symbol') or 'N/A'}",
            f"trade_date: {prompt_context.get('trade_date') or 'N/A'}",
            f"extra_context: {prompt_context.get('extra_context') or 'N/A'}",
            f"knowledge_documents: {prompt_context['knowledge_payload'].get('document_count', 0)}",
            f"available_tools_used: {len(prompt_context.get('tool_results', []))}",
        ]
        return "\n\n".join(
            block
            for block in (
                shared_prompt,
                analyst_prompt,
                "\n".join(context_lines),
            )
            if block
        )

    def synthesize(
        self,
        task: AnalystTask,
        knowledge_payload: dict[str, Any],
        tool_results: list[ToolCallResult],
        prompt: str,
    ) -> dict[str, Any]:
        """Create the analyst result via LLM synthesis or deterministic fallback."""
        if self.llm_client is not None:
            return self._synthesize_with_llm(task, knowledge_payload, tool_results, prompt)
        return self._synthesize_fallback(task, knowledge_payload, tool_results, prompt)

    def _synthesize_with_llm(
        self,
        task: AnalystTask,
        knowledge_payload: dict[str, Any],
        tool_results: list[ToolCallResult],
        prompt: str,
    ) -> dict[str, Any]:
        """Invoke the configured llm_client and normalize the analyst response."""
        llm_payload = self.build_llm_payload(
            task,
            knowledge_payload=knowledge_payload,
            tool_results=tool_results,
        )
        parsed_response = self.llm_client.invoke_json(
            prompt,
            payload=llm_payload,
            schema=self.analyst_output_schema(),
        )
        return self.normalize_llm_result(
            parsed_response,
            task=task,
            knowledge_payload=knowledge_payload,
            tool_results=tool_results,
            prompt=prompt,
        )

    def _synthesize_fallback(
        self,
        task: AnalystTask,
        knowledge_payload: dict[str, Any],
        tool_results: list[ToolCallResult],
        prompt: str,
    ) -> dict[str, Any]:
        """Create a deterministic first-pass analyst result."""
        documents = knowledge_payload.get("documents", [])
        evidence = list(knowledge_payload.get("evidence", []))
        successful_tools = [result for result in tool_results if result.success]
        failed_tools = [result for result in tool_results if not result.success]

        signals = [
            document.get("title", "Untitled evidence")
            for document in documents[:3]
        ]
        if not signals:
            signals = ["No matching knowledge-base signals were found."]

        risks: list[str] = []
        if not documents:
            risks.append("Knowledge coverage is currently thin for this subject.")
        if failed_tools:
            risks.append("One or more analyst tools failed; review tool_trace before acting.")
        if task.symbol and documents and all(
            document.get("metadata", {}).get("symbol") not in (None, "", task.symbol)
            for document in documents
        ):
            risks.append("Retrieved evidence may be thematic rather than symbol-specific.")
        if not risks:
            risks.append("This is a first-pass synthesis pending a model-backed analyst prompt.")

        confidence = "low"
        if documents and successful_tools:
            confidence = "medium"
        if len(documents) >= 3 and len(successful_tools) >= 1:
            confidence = "high"

        summary = (
            f"{self.analyst_name} reviewed {knowledge_payload.get('document_count', 0)} "
            f"knowledge documents for {task.subject}."
        )
        if successful_tools:
            summary += f" {len(successful_tools)} tool call(s) succeeded."
        if task.symbol:
            summary += f" The current symbol focus is {task.symbol}."

        return {
            "analyst": self.analyst_name,
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "summary": summary,
            "signals": signals,
            "risks": risks,
            "confidence": confidence,
            "prompt": prompt,
            "query": knowledge_payload.get("query", ""),
            "documents": documents,
            "evidence": evidence,
            "tool_trace": [self.serialize_tool_result(result) for result in tool_results],
        }

    def build_llm_payload(
        self,
        task: AnalystTask,
        *,
        knowledge_payload: dict[str, Any],
        tool_results: list[ToolCallResult],
    ) -> dict[str, Any]:
        """Build the payload sent into the configured llm_client."""
        return {
            "task": {
                "subject": task.subject,
                "symbol": task.symbol,
                "trade_date": task.trade_date,
                "extra_context": task.extra_context,
            },
            "query": knowledge_payload.get("query", ""),
            "documents": knowledge_payload.get("documents", []),
            "evidence": knowledge_payload.get("evidence", []),
            "tool_trace": [self.serialize_tool_result(result) for result in tool_results],
            "instructions": (
                "Return a JSON object with keys: "
                "summary, signals, risks, confidence, and optional evidence_titles."
            ),
        }

    def normalize_llm_result(
        self,
        llm_result: dict[str, Any],
        *,
        task: AnalystTask,
        knowledge_payload: dict[str, Any],
        tool_results: list[ToolCallResult],
        prompt: str,
    ) -> dict[str, Any]:
        """Merge LLM output with runtime metadata into the analyst result."""
        documents = knowledge_payload.get("documents", [])
        evidence = list(knowledge_payload.get("evidence", []))
        signals = self._normalize_string_list(
            llm_result.get("signals"),
            fallback=[
                document.get("title", "Untitled evidence")
                for document in documents[:3]
            ] or ["No matching knowledge-base signals were found."],
        )
        risks = self._normalize_string_list(
            llm_result.get("risks"),
            fallback=["Model output did not include explicit risks."],
        )
        confidence = str(llm_result.get("confidence", "medium")).strip().lower() or "medium"
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"

        return {
            "analyst": self.analyst_name,
            "subject": task.subject,
            "symbol": task.symbol,
            "trade_date": task.trade_date,
            "summary": str(llm_result.get("summary", "")).strip()
            or f"{self.analyst_name} produced an empty model summary.",
            "signals": signals,
            "risks": risks,
            "confidence": confidence,
            "prompt": prompt,
            "query": knowledge_payload.get("query", ""),
            "documents": documents,
            "evidence": evidence,
            "tool_trace": [self.serialize_tool_result(result) for result in tool_results],
            "raw_model_output": llm_result,
        }

    def analyst_output_schema(self) -> dict[str, Any]:
        """Return the target structured schema for analyst outputs."""
        return {
            "summary": "string",
            "signals": ["string"],
            "risks": ["string"],
            "confidence": "low|medium|high",
            "evidence_titles": ["string"],
        }

    def _normalize_string_list(self, value: Any, *, fallback: list[str]) -> list[str]:
        """Normalize a model field into a list of strings."""
        if isinstance(value, list):
            normalized = [str(item).strip() for item in value if str(item).strip()]
            return normalized or fallback
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return fallback

    def invoke(self, task: AnalystTask) -> dict[str, Any]:
        """Run the analyst end-to-end for a single task."""
        tool_calls = self.plan_tool_calls(task)
        tool_results = self.run_tools(tool_calls)
        knowledge_payload = self.retrieve_knowledge(task, tool_results=tool_results)
        prompt_context = self.build_prompt_context(task, knowledge_payload, tool_results)
        prompt = self.render_prompt(prompt_context)
        return self.synthesize(task, knowledge_payload, tool_results, prompt)

    def as_node(self) -> Any:
        """Expose the analyst as a LangGraph-compatible node function."""

        def node(state: AnalystRuntimeState) -> AnalystRuntimeState:
            task = AnalystTask.from_state(state)
            result = self.invoke(task)
            analyst_outputs = dict(state.get("analyst_outputs", {}))
            analyst_outputs[self.analyst_name] = result
            messages = list(state.get("messages", []))
            messages.append(self.build_agent_message(result))
            return {
                "messages": messages,
                "analyst_outputs": analyst_outputs,
            }

        return node

    def build_agent_message(self, result: dict[str, Any]) -> Any:
        """Create a message payload that can live in graph state."""
        content = result.get("summary", "")
        try:
            from langchain_core.messages import AIMessage
        except ModuleNotFoundError:
            return {
                "role": "assistant",
                "name": self.analyst_name,
                "content": content,
            }
        return AIMessage(content=content, name=self.analyst_name)

    def serialize_tool_result(self, result: ToolCallResult) -> dict[str, Any]:
        """Convert a tool result into a JSON-friendly payload."""
        return {
            "tool_name": result.tool_name,
            "success": result.success,
            "content": result.content,
            "structured_data": result.structured_data,
            "error": result.error,
        }
