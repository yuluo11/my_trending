"""Application entrypoints for analyst realization and analyst-runtime assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from .knowledge.repository import DatasetName, KnowledgeRepository
from .llm.client import LLMClient, LLMRunnable, ensure_llm_client
from .agents.analysts.base_agent import (
    AnalystRuntimeState,
    AnalystTask,
    FilePromptProvider as AnalystFilePromptProvider,
    BaseLangGraphAnalystAgent,
)
from .agents.decision import DecisionAdvisoryAgent, DecisionTask
from .agents.decision.base_agent import FilePromptProvider as DecisionFilePromptProvider
from .agents.analysts.graph_agent import GraphAnalystAgent
from .agents.analysts.market_agent import MarketAnalystAgent
from .agents.analysts.news_agent import NewsAnalystAgent
from .agents.analysts.orchestrator import AnalystOrchestrator
from .agents.analysts.sentiment_agent import SentimentAnalystAgent
from .agents.analysts.social_agent import SocialAnalystAgent
from .services.analysts.graph_service import GraphAnalystService
from .services.analysts.market_service import MarketAnalystService
from .services.analysts.news_service import NewsAnalystService
from .services.analysts.sentiment_service import SentimentAnalystService
from .services.analysts.social_service import SocialAnalystService
from .services.decision import DecisionKnowledgeService
from .tools.analyst.tooling import AnalystToolRegistry, KnowledgeBaseSearchTool

SRC_DIR = Path(__file__).resolve().parent
PROMPTS_ROOT_DIR = SRC_DIR / "prompts"
ANALYST_PROMPTS_DIR = PROMPTS_ROOT_DIR / "analysts"
DECISION_PROMPTS_DIR = PROMPTS_ROOT_DIR / "decision"
DEFAULT_ANALYST_SEQUENCE = (
    "market_analyst",
    "news_analyst",
    "sentiment_analyst",
    "social_analyst",
    "graph_analyst",
)


class AppRuntimeState(TypedDict, total=False):
    """Top-level workflow state used by the first LangGraph runtime."""

    subject: str
    symbol: str | None
    trade_date: str | None
    extra_context: str | None
    datasets: tuple[DatasetName, ...] | list[DatasetName] | None
    metadata_filter: dict[str, Any] | None
    max_documents: int | None
    messages: list[Any]
    analyst_outputs: dict[str, dict[str, Any]]
    decision_output: dict[str, Any]
    portfolio_context: dict[str, Any] | None


def build_prompt_provider(
    prompts_dir: str | Path | None = None,
) -> AnalystFilePromptProvider:
    """Build the prompt provider backed by prompt files on disk."""
    return AnalystFilePromptProvider(prompts_dir or ANALYST_PROMPTS_DIR)


def build_tool_registry(
    service: Any,
) -> AnalystToolRegistry:
    """Register the first analyst tools available to the runtime."""
    registry = AnalystToolRegistry()
    registry.register(KnowledgeBaseSearchTool(service))
    return registry


def build_decision_prompt_provider(
    prompts_dir: str | Path | None = None,
) -> DecisionFilePromptProvider:
    """Build the prompt provider for the decision advisory layer."""
    return DecisionFilePromptProvider(prompts_dir or DECISION_PROMPTS_DIR)


def build_graph_analyst_agent(
    *,
    repository: KnowledgeRepository | None = None,
    prompt_provider: AnalystFilePromptProvider | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> GraphAnalystAgent:
    """Assemble the graph analyst agent."""
    service = GraphAnalystService(repository=repository)
    registry = build_tool_registry(service)
    return GraphAnalystAgent(
        service=service,
        tool_registry=registry,
        prompt_provider=prompt_provider or build_prompt_provider(),
        llm_client=ensure_llm_client(llm_client=llm_client, llm=llm),
    )


def build_market_analyst_agent(
    *,
    repository: KnowledgeRepository | None = None,
    prompt_provider: AnalystFilePromptProvider | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> MarketAnalystAgent:
    """Assemble the market analyst agent."""
    service = MarketAnalystService(repository=repository)
    registry = build_tool_registry(service)
    return MarketAnalystAgent(
        service=service,
        tool_registry=registry,
        prompt_provider=prompt_provider or build_prompt_provider(),
        llm_client=ensure_llm_client(llm_client=llm_client, llm=llm),
    )


def build_news_analyst_agent(
    *,
    repository: KnowledgeRepository | None = None,
    prompt_provider: AnalystFilePromptProvider | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> NewsAnalystAgent:
    """Assemble the news analyst agent."""
    service = NewsAnalystService(repository=repository)
    registry = build_tool_registry(service)
    return NewsAnalystAgent(
        service=service,
        tool_registry=registry,
        prompt_provider=prompt_provider or build_prompt_provider(),
        llm_client=ensure_llm_client(llm_client=llm_client, llm=llm),
    )


def build_sentiment_analyst_agent(
    *,
    repository: KnowledgeRepository | None = None,
    prompt_provider: AnalystFilePromptProvider | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> SentimentAnalystAgent:
    """Assemble the sentiment analyst agent."""
    service = SentimentAnalystService(repository=repository)
    registry = build_tool_registry(service)
    return SentimentAnalystAgent(
        service=service,
        tool_registry=registry,
        prompt_provider=prompt_provider or build_prompt_provider(),
        llm_client=ensure_llm_client(llm_client=llm_client, llm=llm),
    )


def build_social_analyst_agent(
    *,
    repository: KnowledgeRepository | None = None,
    prompt_provider: AnalystFilePromptProvider | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> SocialAnalystAgent:
    """Assemble the social analyst agent."""
    service = SocialAnalystService(repository=repository)
    registry = build_tool_registry(service)
    return SocialAnalystAgent(
        service=service,
        tool_registry=registry,
        prompt_provider=prompt_provider or build_prompt_provider(),
        llm_client=ensure_llm_client(llm_client=llm_client, llm=llm),
    )


def build_default_analyst_agents(
    *,
    repository: KnowledgeRepository | None = None,
    prompt_provider: AnalystFilePromptProvider | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> dict[str, BaseLangGraphAnalystAgent]:
    """Build the default set of analyst agents used by the workflow."""
    resolved_prompt_provider = prompt_provider or build_prompt_provider()
    return {
        "market_analyst": build_market_analyst_agent(
            repository=repository,
            prompt_provider=resolved_prompt_provider,
            llm_client=llm_client,
            llm=llm,
        ),
        "news_analyst": build_news_analyst_agent(
            repository=repository,
            prompt_provider=resolved_prompt_provider,
            llm_client=llm_client,
            llm=llm,
        ),
        "sentiment_analyst": build_sentiment_analyst_agent(
            repository=repository,
            prompt_provider=resolved_prompt_provider,
            llm_client=llm_client,
            llm=llm,
        ),
        "social_analyst": build_social_analyst_agent(
            repository=repository,
            prompt_provider=resolved_prompt_provider,
            llm_client=llm_client,
            llm=llm,
        ),
        "graph_analyst": build_graph_analyst_agent(
            repository=repository,
            prompt_provider=resolved_prompt_provider,
            llm_client=llm_client,
            llm=llm,
        ),
    }


def build_analyst_orchestrator(
    *,
    repository: KnowledgeRepository | None = None,
    prompt_provider: AnalystFilePromptProvider | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> AnalystOrchestrator:
    """Build the internal orchestrator coordinating the default analyst sequence."""
    return AnalystOrchestrator(
        analysts=build_default_analyst_agents(
            repository=repository,
            prompt_provider=prompt_provider,
            llm_client=llm_client,
            llm=llm,
        ),
        sequence=DEFAULT_ANALYST_SEQUENCE,
        llm_client=ensure_llm_client(llm_client=llm_client, llm=llm),
        prompts_dir=ANALYST_PROMPTS_DIR,
    )


def build_decision_advisory_agent(
    *,
    repository: KnowledgeRepository | None = None,
    prompt_provider: DecisionFilePromptProvider | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> DecisionAdvisoryAgent:
    """Assemble the decision advisory agent and its dedicated prompt assets."""
    service = DecisionKnowledgeService(repository=repository)
    return DecisionAdvisoryAgent(
        service=service,
        prompt_provider=prompt_provider or build_decision_prompt_provider(),
        llm_client=ensure_llm_client(llm_client=llm_client, llm=llm),
    )


def run_graph_analyst(
    *,
    subject: str,
    symbol: str | None = None,
    trade_date: str | None = None,
    extra_context: str | None = None,
    datasets: tuple[DatasetName, ...] | None = None,
    metadata_filter: dict[str, Any] | None = None,
    max_documents: int | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> dict[str, Any]:
    """Run the graph analyst directly without compiling a LangGraph workflow."""
    task = AnalystTask(
        subject=subject,
        symbol=symbol,
        trade_date=trade_date,
        extra_context=extra_context,
        datasets=datasets,
        metadata_filter=metadata_filter,
        max_documents=max_documents,
    )
    agent = build_graph_analyst_agent(llm_client=llm_client, llm=llm)
    return agent.invoke(task)


def run_market_analyst(
    *,
    subject: str,
    symbol: str | None = None,
    trade_date: str | None = None,
    extra_context: str | None = None,
    datasets: tuple[DatasetName, ...] | None = None,
    metadata_filter: dict[str, Any] | None = None,
    max_documents: int | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> dict[str, Any]:
    """Run the market analyst directly."""
    task = AnalystTask(
        subject=subject,
        symbol=symbol,
        trade_date=trade_date,
        extra_context=extra_context,
        datasets=datasets,
        metadata_filter=metadata_filter,
        max_documents=max_documents,
    )
    return build_market_analyst_agent(llm_client=llm_client, llm=llm).invoke(task)


def run_news_analyst(
    *,
    subject: str,
    symbol: str | None = None,
    trade_date: str | None = None,
    extra_context: str | None = None,
    datasets: tuple[DatasetName, ...] | None = None,
    metadata_filter: dict[str, Any] | None = None,
    max_documents: int | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> dict[str, Any]:
    """Run the news analyst directly."""
    task = AnalystTask(
        subject=subject,
        symbol=symbol,
        trade_date=trade_date,
        extra_context=extra_context,
        datasets=datasets,
        metadata_filter=metadata_filter,
        max_documents=max_documents,
    )
    return build_news_analyst_agent(llm_client=llm_client, llm=llm).invoke(task)


def run_sentiment_analyst(
    *,
    subject: str,
    symbol: str | None = None,
    trade_date: str | None = None,
    extra_context: str | None = None,
    datasets: tuple[DatasetName, ...] | None = None,
    metadata_filter: dict[str, Any] | None = None,
    max_documents: int | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> dict[str, Any]:
    """Run the sentiment analyst directly."""
    task = AnalystTask(
        subject=subject,
        symbol=symbol,
        trade_date=trade_date,
        extra_context=extra_context,
        datasets=datasets,
        metadata_filter=metadata_filter,
        max_documents=max_documents,
    )
    return build_sentiment_analyst_agent(llm_client=llm_client, llm=llm).invoke(task)


def run_social_analyst(
    *,
    subject: str,
    symbol: str | None = None,
    trade_date: str | None = None,
    extra_context: str | None = None,
    datasets: tuple[DatasetName, ...] | None = None,
    metadata_filter: dict[str, Any] | None = None,
    max_documents: int | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> dict[str, Any]:
    """Run the social analyst directly."""
    task = AnalystTask(
        subject=subject,
        symbol=symbol,
        trade_date=trade_date,
        extra_context=extra_context,
        datasets=datasets,
        metadata_filter=metadata_filter,
        max_documents=max_documents,
    )
    return build_social_analyst_agent(llm_client=llm_client, llm=llm).invoke(task)


def run_analyst_orchestrator(
    *,
    subject: str,
    symbol: str | None = None,
    trade_date: str | None = None,
    extra_context: str | None = None,
    datasets: tuple[DatasetName, ...] | None = None,
    metadata_filter: dict[str, Any] | None = None,
    max_documents: int | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> dict[str, Any]:
    """Run the internal multi-analyst orchestrator and return the aggregate result."""
    task = AnalystTask(
        subject=subject,
        symbol=symbol,
        trade_date=trade_date,
        extra_context=extra_context,
        datasets=datasets,
        metadata_filter=metadata_filter,
        max_documents=max_documents,
    )
    orchestrator = build_analyst_orchestrator(llm_client=llm_client, llm=llm)
    return orchestrator.run(task)


def run_analyst_realization(
    *,
    subject: str,
    symbol: str | None = None,
    trade_date: str | None = None,
    extra_context: str | None = None,
    datasets: tuple[DatasetName, ...] | None = None,
    metadata_filter: dict[str, Any] | None = None,
    max_documents: int | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> dict[str, Any]:
    """Run a full analyst realization and return the top-level aggregated result."""
    return run_analyst_orchestrator(
        subject=subject,
        symbol=symbol,
        trade_date=trade_date,
        extra_context=extra_context,
        datasets=datasets,
        metadata_filter=metadata_filter,
        max_documents=max_documents,
        llm_client=llm_client,
        llm=llm,
    )


def run_decision_advisory(
    *,
    analyst_payload: dict[str, Any],
    portfolio_context: dict[str, Any] | None = None,
    datasets: tuple[DatasetName, ...] | None = None,
    metadata_filter: dict[str, Any] | None = None,
    max_documents: int | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> dict[str, Any]:
    """Run the decision advisory agent over an analyst orchestration payload."""
    task = DecisionTask.from_analyst_payload(
        analyst_payload,
        portfolio_context=portfolio_context,
        datasets=datasets,
        metadata_filter=metadata_filter,
        max_documents=max_documents,
    )
    agent = build_decision_advisory_agent(llm_client=llm_client, llm=llm)
    return agent.invoke(task)


def run_decision_realization(
    *,
    subject: str,
    symbol: str | None = None,
    trade_date: str | None = None,
    extra_context: str | None = None,
    portfolio_context: dict[str, Any] | None = None,
    analyst_datasets: tuple[DatasetName, ...] | None = None,
    analyst_metadata_filter: dict[str, Any] | None = None,
    analyst_max_documents: int | None = None,
    decision_datasets: tuple[DatasetName, ...] | None = None,
    decision_metadata_filter: dict[str, Any] | None = None,
    decision_max_documents: int | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> dict[str, Any]:
    """Run the analyst layer first, then synthesize an advisory decision payload."""
    analyst_payload = run_analyst_orchestrator(
        subject=subject,
        symbol=symbol,
        trade_date=trade_date,
        extra_context=extra_context,
        datasets=analyst_datasets,
        metadata_filter=analyst_metadata_filter,
        max_documents=analyst_max_documents,
        llm_client=llm_client,
        llm=llm,
    )
    decision_payload = run_decision_advisory(
        analyst_payload=analyst_payload,
        portfolio_context=portfolio_context,
        datasets=decision_datasets,
        metadata_filter=decision_metadata_filter,
        max_documents=decision_max_documents,
        llm_client=llm_client,
        llm=llm,
    )
    return {
        "subject": subject,
        "symbol": symbol,
        "trade_date": trade_date,
        "portfolio_context": portfolio_context,
        "analyst": analyst_payload,
        "decision": decision_payload,
    }


def build_langgraph_workflow(
    *,
    repository: KnowledgeRepository | None = None,
    prompt_provider: AnalystFilePromptProvider | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> Any:
    """Compile the default multi-analyst workflow when optional deps are installed."""
    try:
        from langgraph.graph import END, StateGraph
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "langgraph is not installed. Install langgraph to compile the analyst runtime."
        ) from error

    agents = build_default_analyst_agents(
        repository=repository,
        prompt_provider=prompt_provider,
        llm_client=llm_client,
        llm=llm,
    )
    workflow = StateGraph(AppRuntimeState)
    for analyst_name in DEFAULT_ANALYST_SEQUENCE:
        workflow.add_node(analyst_name, agents[analyst_name].as_node())
    workflow.set_entry_point(DEFAULT_ANALYST_SEQUENCE[0])
    for current_name, next_name in zip(
        DEFAULT_ANALYST_SEQUENCE,
        DEFAULT_ANALYST_SEQUENCE[1:],
    ):
        workflow.add_edge(current_name, next_name)
    workflow.add_edge(DEFAULT_ANALYST_SEQUENCE[-1], END)
    return workflow.compile()


def run_langgraph_workflow(
    *,
    subject: str,
    symbol: str | None = None,
    trade_date: str | None = None,
    extra_context: str | None = None,
    datasets: tuple[DatasetName, ...] | None = None,
    metadata_filter: dict[str, Any] | None = None,
    max_documents: int | None = None,
    llm_client: LLMClient | None = None,
    llm: LLMRunnable | None = None,
) -> AnalystRuntimeState:
    """Run the compiled workflow when langgraph is available."""
    workflow = build_langgraph_workflow(llm_client=llm_client, llm=llm)
    initial_state: AnalystRuntimeState = {
        "subject": subject,
        "symbol": symbol,
        "trade_date": trade_date,
        "extra_context": extra_context,
        "datasets": datasets,
        "metadata_filter": metadata_filter,
        "max_documents": max_documents,
        "messages": [],
        "analyst_outputs": {},
    }
    return workflow.invoke(initial_state)
