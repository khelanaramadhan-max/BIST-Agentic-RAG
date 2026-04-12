"""
Agent tools – LangChain retriever tools wrapping each ChromaDB collection
plus live web search for general queries.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from langchain_core.documents import Document

from langchain_community.tools import DuckDuckGoSearchRun
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
    try:
        retriever = get_retriever("kap_disclosure", ticker=ticker or None)
        docs = retriever.invoke(query)
        logger.info("KAP retrieval: %d docs for query='%s'", len(docs), query[:60])
        return _format_docs(docs)
    except Exception as exc:
        logger.warning("KAP retrieval error: %s", exc)
        return "No documents found."


@tool
def search_financial_news(query: str, ticker: str = "") -> str:
    """
    Search Turkish financial news articles from Bloomberg HT, Dünya Gazetesi,
    and other financial media sources.
    Use this for market narratives, sentiment analysis, sector developments,
    and media framing of company events.
    """
    try:
        retriever = get_retriever("news", ticker=ticker or None)
        docs = retriever.invoke(query)
        logger.info("News retrieval: %d docs for query='%s'", len(docs), query[:60])
        return _format_docs(docs)
    except Exception as exc:
        logger.warning("News retrieval error: %s", exc)
        return "No documents found."


@tool
def search_brokerage_reports(query: str, ticker: str = "") -> str:
    """
    Search brokerage equity research reports from Turkish investment banks.
    Use this for qualitative analyst commentary, sector theme analysis,
    and narrative evolution over time. Never use for price targets or recommendations.
    """
    try:
        retriever = get_retriever("brokerage_report", ticker=ticker or None)
        docs = retriever.invoke(query)
        logger.info("Brokerage retrieval: %d docs for query='%s'", len(docs), query[:60])
        return _format_docs(docs)
    except Exception as exc:
        logger.warning("Brokerage retrieval error: %s", exc)
        return "No documents found."


@tool
def search_all_sources(query: str, ticker: str = "") -> str:
    """
    Search across all available sources (KAP, news, brokerage reports) simultaneously.
    Use when the question spans multiple source types or you are unsure which source
    is most relevant.
    """
    results = {}
    for source in ["kap_disclosure", "news", "brokerage_report"]:
        try:
            retriever = get_retriever(source, ticker=ticker or None, k=3)
            docs = retriever.invoke(query)
            results[source] = docs
        except Exception as exc:
            logger.warning("Multi-source retrieval error (%s): %s", source, exc)
            results[source] = []

    combined = []
    for source, docs in results.items():
        if docs:
            combined.append(f"=== {source.upper()} ===")
            combined.append(_format_docs(docs))

    return "\n\n".join(combined) if combined else "No documents found across any source."


@tool
def search_live_web(query: str, ticker: str = "") -> str:
    """
    Search the live web using DuckDuckGo for the most recent, up-to-date information.
    Use this for ANY topic: current events, general knowledge, coding, science,
    finance, sports, history, or anything the user asks about.
    This is the most versatile tool — it can find information on virtually any subject.
    """
    search = DuckDuckGoSearchRun()
    # Build a smart search query
    if ticker and any(kw in query.lower() for kw in ["borsa", "bist", "stock", "share", "hisse"]):
        q = f"{ticker} Borsa Istanbul {query}"
    elif ticker:
        q = f"{ticker} {query}"
    else:
        q = query
    
    logger.info("Executing Live Web Search for query: '%s'", q)
    try:
        results = search.invoke(q)
        if results and len(results.strip()) > 10:
            return f"[Web Search Results]\n{results}"
        else:
            return "No relevant web results found."
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return "Web search unavailable at the moment."


# All tools as a list for the agent
ALL_TOOLS = [
    search_kap_disclosures,
    search_financial_news,
    search_brokerage_reports,
    search_all_sources,
    search_live_web,
]
