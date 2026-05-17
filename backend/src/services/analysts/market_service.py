"""Market analyst service backed by the shared knowledge layer."""

from .graph_service import KnowledgeBackedAnalystService


class MarketAnalystService(KnowledgeBackedAnalystService):
    """Retrieve strategy and market context for broader market analysis."""

    analyst_name = "market_analyst"
    default_datasets = ("foundation", "dynamic")
