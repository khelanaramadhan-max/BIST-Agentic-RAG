"""
Agent tools – LangChain retriever tools wrapping each ChromaDB collection.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from langchain_core.documents import Document

from vectordb.chroma_store import get_retriever

logger = logging.getLogger(__name__)


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved docs into a readable context string with citations."""
    if not docs:
        return "No documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        citation = (
            f"[Source {i}: {meta.get('institution', 'Unknown')} | "
            f"Type: {meta.get('source_type', '?')} | "
            f"Date: {meta.get('date', '?')} | "
            f"Ticker: {meta.get('ticker', '?')}]"
        )
        parts.append(f"{citation}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


@tool
def search_kap_disclosures(query: str, ticker: str = "") -> str:
    """
    Search official KAP (Kamu Aydınlatma Platformu) disclosures.
    Use this for authoritative company disclosures, board decisions,
    material events, and regulatory filings.
    KAP is the ground-truth source — treat its content as authoritative.
    """
    retriever = get_retriever("kap_disclosure", ticker=ticker or None)
    docs = retriever.invoke(query)
    logger.info("KAP retrieval: %d docs for query='%s'", len(docs), query[:60])
    return _format_docs(docs)


@tool
def search_financial_news(query: str, ticker: str = "") -> str:
    """
    Search Turkish financial news articles from Bloomberg HT, Dünya Gazetesi,
    and other financial media sources.
    Use this for market narratives, sentiment analysis, sector developments,
    and media framing of company events.
    """
    retriever = get_retriever("news", ticker=ticker or None)
    docs = retriever.invoke(query)
    logger.info("News retrieval: %d docs for query='%s'", len(docs), query[:60])
    return _format_docs(docs)


@tool
def search_brokerage_reports(query: str, ticker: str = "") -> str:
    """
    Search brokerage equity research reports from Turkish investment banks.
    Use this for qualitative analyst commentary, sector theme analysis,
    and narrative evolution over time. Never use for price targets or recommendations.
    """
    retriever = get_retriever("brokerage_report", ticker=ticker or None)
    docs = retriever.invoke(query)
    logger.info("Brokerage retrieval: %d docs for query='%s'", len(docs), query[:60])
    return _format_docs(docs)


@tool
def search_all_sources(query: str, ticker: str = "") -> str:
    """
    Search across all available sources (KAP, news, brokerage reports) simultaneously.
    Use when the question spans multiple source types or you are unsure which source
    is most relevant.
    """
    results = {}
    for source in ["kap_disclosure", "news", "brokerage_report"]:
        retriever = get_retriever(source, ticker=ticker or None, k=3)
        docs = retriever.invoke(query)
        results[source] = docs

    combined = []
    for source, docs in results.items():
        if docs:
            combined.append(f"=== {source.upper()} ===")
            combined.append(_format_docs(docs))

    return "\n\n".join(combined) if combined else "No documents found across any source."


# All tools as a list for the agent
ALL_TOOLS = [
    search_kap_disclosures,
    search_financial_news,
    search_brokerage_reports,
    search_all_sources,
]
