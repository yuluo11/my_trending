from __future__ import annotations

import unittest
from typing import Any

from backend.src.agents.analysts.base_agent import (
    AnalystTask,
    BaseLangGraphAnalystAgent,
    StaticPromptProvider,
)
from backend.src.agents.analysts.orchestrator import AnalystOrchestrator
from backend.src.tools.analyst.tooling import AnalystToolRegistry, ToolCallResult


class FakeKnowledgeService:
    analyst_name = "market_analyst"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def analyze(
        self,
        subject: str,
        *,
        extra_context: str | None = None,
        datasets: tuple[str, ...] | None = None,
        symbol: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "subject": subject,
                "extra_context": extra_context,
                "datasets": datasets,
                "symbol": symbol,
                "metadata_filter": metadata_filter,
                "k": k,
            }
        )
        return self.payload


class SuccessfulSearchTool:
    name = "search_knowledge_base"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def invoke(self, request: Any) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            success=True,
            content="Retrieved 1 knowledge document.",
            structured_data=self.payload,
        )


class FailingSearchTool:
    name = "search_knowledge_base"

    def invoke(self, request: Any) -> ToolCallResult:
        return ToolCallResult(
            tool_name=self.name,
            success=False,
            content="Knowledge-base search failed.",
            error="simulated tool failure",
        )


class StubAnalyst:
    def __init__(self, analyst_name: str, result: dict[str, Any]) -> None:
        self.analyst_name = analyst_name
        self.result = result
        self.seen_messages: list[list[Any]] = []

    def invoke(self, task: AnalystTask) -> dict[str, Any]:
        self.seen_messages.append(list(task.messages))
        return dict(self.result)

    def build_agent_message(self, result: dict[str, Any]) -> dict[str, str]:
        return {
            "role": "assistant",
            "name": self.analyst_name,
            "content": result["summary"],
        }


class AnalystRuntimeTests(unittest.TestCase):
    def build_agent(
        self,
        *,
        payload: dict[str, Any],
        tool: Any | None = None,
    ) -> tuple[BaseLangGraphAnalystAgent, FakeKnowledgeService]:
        service = FakeKnowledgeService(payload)
        registry = AnalystToolRegistry()
        if tool is not None:
            registry.register(tool)

        agent = BaseLangGraphAnalystAgent(
            analyst_name="market_analyst",
            knowledge_service=service,
            tool_registry=registry,
            prompt_provider=StaticPromptProvider(
                shared_prompt="shared prompt",
                analyst_prompts={"market_analyst": "role prompt"},
            ),
        )
        return agent, service

    def test_agent_uses_tool_payload_without_falling_back_to_service(self) -> None:
        payload = {
            "query": "NVIDIA constructive reset",
            "document_count": 1,
            "documents": [
                {
                    "title": "Sample Strategy Record",
                    "text": "Trend-following setups should prioritize risk control first.",
                    "metadata": {"symbol": "NVDA", "category": "strategy"},
                }
            ],
            "evidence": [{"title": "Sample Strategy Record"}],
        }
        agent, service = self.build_agent(
            payload=payload,
            tool=SuccessfulSearchTool(payload),
        )

        result = agent.invoke(
            AnalystTask(
                subject="NVIDIA constructive reset",
                symbol="NVDA",
                datasets=("foundation", "dynamic"),
                max_documents=3,
            )
        )

        self.assertEqual([], service.calls)
        self.assertEqual("market_analyst", result["analyst"])
        self.assertEqual("medium", result["confidence"])
        self.assertEqual(["Sample Strategy Record"], result["signals"])
        self.assertEqual(1, len(result["tool_trace"]))
        self.assertIn("1 tool call(s) succeeded.", result["summary"])
        self.assertIn("knowledge_documents: 1", result["prompt"])

    def test_agent_falls_back_to_service_and_reports_tool_failure(self) -> None:
        payload = {
            "query": "NVIDIA constructive reset",
            "document_count": 0,
            "documents": [],
            "evidence": [],
        }
        agent, service = self.build_agent(
            payload=payload,
            tool=FailingSearchTool(),
        )

        result = agent.invoke(
            AnalystTask(
                subject="NVIDIA constructive reset",
                symbol="NVDA",
                datasets=("foundation",),
                metadata_filter={"category": "strategy"},
                max_documents=2,
            )
        )

        self.assertEqual(1, len(service.calls))
        self.assertEqual("low", result["confidence"])
        self.assertIn("Knowledge coverage is currently thin", " ".join(result["risks"]))
        self.assertIn("One or more analyst tools failed", " ".join(result["risks"]))
        self.assertEqual("simulated tool failure", result["tool_trace"][0]["error"])

    def test_as_node_updates_state_with_output_and_message(self) -> None:
        payload = {
            "query": "NVIDIA constructive reset",
            "document_count": 1,
            "documents": [
                {
                    "title": "Sample Strategy Record",
                    "text": "Trend-following setups should prioritize risk control first.",
                    "metadata": {"symbol": "NVDA", "category": "strategy"},
                }
            ],
            "evidence": [{"title": "Sample Strategy Record"}],
        }
        agent, _ = self.build_agent(payload=payload, tool=SuccessfulSearchTool(payload))

        node = agent.as_node()
        state = node(
            {
                "subject": "NVIDIA constructive reset",
                "symbol": "NVDA",
                "messages": [{"role": "system", "content": "existing"}],
                "analyst_outputs": {},
            }
        )

        self.assertIn("market_analyst", state["analyst_outputs"])
        self.assertEqual(2, len(state["messages"]))
        message = state["messages"][-1]
        if isinstance(message, dict):
            self.assertEqual("market_analyst", message["name"])
            self.assertTrue(message["content"])
        else:
            self.assertEqual("market_analyst", getattr(message, "name"))
            self.assertTrue(getattr(message, "content"))

    def test_orchestrator_aggregates_results_and_threads_messages(self) -> None:
        analyst_results = {
            "market_analyst": {
                "analyst": "market_analyst",
                "summary": "Market setup looks constructive.",
                "signals": ["Trend support"],
                "risks": ["Valuation risk"],
                "confidence": "high",
                "documents": [{"title": "Sample Strategy Record"}],
            },
            "news_analyst": {
                "analyst": "news_analyst",
                "summary": "News flow is mixed.",
                "signals": ["Headline support"],
                "risks": ["Catalyst uncertainty"],
                "confidence": "medium",
                "documents": [],
            },
        }
        analysts = {
            name: StubAnalyst(name, result)
            for name, result in analyst_results.items()
        }
        orchestrator = AnalystOrchestrator(
            analysts=analysts,
            sequence=("market_analyst", "news_analyst"),
        )

        result = orchestrator.run(
            AnalystTask(
                subject="NVIDIA constructive reset",
                symbol="NVDA",
                messages=[{"role": "system", "content": "seed"}],
            )
        )

        self.assertEqual("high", result["overall_confidence"])
        self.assertEqual(
            ["Trend support", "Headline support"],
            result["key_signals"],
        )
        self.assertEqual(
            ["Valuation risk", "Catalyst uncertainty"],
            result["portfolio_risks"],
        )
        self.assertEqual(3, result["message_count"])
        self.assertEqual(1, len(analysts["market_analyst"].seen_messages[0]))
        self.assertEqual(2, len(analysts["news_analyst"].seen_messages[0]))
        self.assertTrue(result["cross_analyst_observations"])


if __name__ == "__main__":
    unittest.main()
