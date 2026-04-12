"""
ChromaDB store – retriever factory and collection management.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import settings
from ingestion.embedder import get_embedding_model, COLLECTION_MAP

logger = logging.getLogger(__name__)

SourceType = Literal["kap_disclosure", "news", "brokerage_report", "all"]


def get_retriever(
    source_type: SourceType = "all",
    ticker: str | None = None,
    k: int | None = None,
) -> VectorStoreRetriever:
    """
    Return a LangChain retriever for the given source type.

    Args:
        source_type: Which collection to search ("kap_disclosure", "news",
                     "brokerage_report", or "all").
        ticker: If provided, adds a metadata filter for that ticker.
        k: Number of documents to retrieve (defaults to settings.top_k_retrieval).
    """
    k = k or settings.top_k_retrieval
    embed = get_embedding_model()

    if source_type == "all":
        # Merge all collections by returning the highest-scoring collection
        # (simple approach: use general chromadb with metadata filter)
        store = Chroma(
            collection_name="kap_disclosures",
            embedding_function=embed,
            persist_directory=settings.chroma_persist_dir,
        )
    else:
        collection = COLLECTION_MAP.get(source_type, "kap_disclosures")
        store = Chroma(
            collection_name=collection,
            embedding_function=embed,
            persist_directory=settings.chroma_persist_dir,
        )

    search_kwargs: dict = {"k": k}
    if ticker:
        search_kwargs["filter"] = {"ticker": ticker}

    return store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def get_multi_retriever(
    ticker: str | None = None,
    k_per_source: int = 3,
) -> dict[str, VectorStoreRetriever]:
    """
    Return a dict of retrievers, one per source type.
    Used by the agentic loop for parallel source querying.
    """
    return {
        "kap": get_retriever("kap_disclosure", ticker=ticker, k=k_per_source),
        "news": get_retriever("news", ticker=ticker, k=k_per_source),
        "brokerage": get_retriever("brokerage_report", ticker=ticker, k=k_per_source),
    }


def collection_stats() -> dict[str, int]:
    """Return document counts for all collections."""
    embed = get_embedding_model()
    stats = {}
    for source_type, coll_name in COLLECTION_MAP.items():
        try:
            store = Chroma(
                collection_name=coll_name,
                embedding_function=embed,
                persist_directory=settings.chroma_persist_dir,
            )
            stats[coll_name] = store._collection.count()
        except Exception:
            stats[coll_name] = 0
    return stats
