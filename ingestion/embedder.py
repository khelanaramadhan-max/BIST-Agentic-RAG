"""
Embedder – chunks documents and upserts them into ChromaDB.
Uses a multilingual sentence-transformer for Turkish language support.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)


# ─── Embedding Model (singleton) ─────────────────────────────────────────────

_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


# ─── ChromaDB Collections ────────────────────────────────────────────────────

COLLECTION_MAP = {
    "kap_disclosure": "kap_disclosures",
    "news": "financial_news",
    "brokerage_report": "brokerage_reports",
}

_stores: dict[str, Chroma] = {}


def get_vector_store(source_type: str) -> Chroma:
    """Return (or create) a ChromaDB collection for a given source type."""
    collection_name = COLLECTION_MAP.get(source_type, "general")
    if collection_name not in _stores:
        _stores[collection_name] = Chroma(
            collection_name=collection_name,
            embedding_function=get_embedding_model(),
            persist_directory=settings.chroma_persist_dir,
        )
        logger.info("Opened ChromaDB collection: %s", collection_name)
    return _stores[collection_name]


# ─── Ingest ──────────────────────────────────────────────────────────────────


def _docs_to_langchain(raw_docs: list[dict[str, Any]]) -> list[Document]:
    """Convert raw doc dicts to LangChain Document objects."""
    lc_docs = []
    for doc in raw_docs:
        content = doc.get("content", "")
        if not content.strip():
            continue
        metadata = {
            "ticker": doc.get("ticker", ""),
            "source_type": doc.get("source_type", "unknown"),
            "date": doc.get("date", ""),
            "institution": doc.get("institution", ""),
            "title": doc.get("title", "")[:200],
            "url": doc.get("url", ""),
            "chunk_index": doc.get("chunk_index", 0),
        }
        lc_docs.append(Document(page_content=content, metadata=metadata))
    return lc_docs


def ingest_documents(raw_docs: list[dict[str, Any]]) -> int:
    """
    Embed and upsert a list of raw documents into the appropriate ChromaDB collection.
    Returns the number of documents successfully ingested.
    """
    if not raw_docs:
        logger.warning("No documents to ingest.")
        return 0

    # Group by source_type
    groups: dict[str, list[dict]] = {}
    for doc in raw_docs:
        st = doc.get("source_type", "unknown")
        groups.setdefault(st, []).append(doc)

    total = 0
    for source_type, docs in groups.items():
        lc_docs = _docs_to_langchain(docs)
        if not lc_docs:
            continue
        store = get_vector_store(source_type)

        # Generate unique IDs to avoid duplicates on re-run
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, d.page_content[:100])) for d in lc_docs]

        # Batch insert (ChromaDB handles duplicates by ID)
        BATCH = 100
        for i in range(0, len(lc_docs), BATCH):
            batch_docs = lc_docs[i : i + BATCH]
            batch_ids = ids[i : i + BATCH]
            try:
                store.add_documents(batch_docs, ids=batch_ids)
                total += len(batch_docs)
            except Exception as exc:
                logger.warning("Batch insert error (source=%s): %s", source_type, exc)

        logger.info(
            "Ingested %d docs into collection '%s'",
            len(lc_docs),
            COLLECTION_MAP.get(source_type, "general"),
        )

    return total


# ─── Load from JSON files ────────────────────────────────────────────────────


def load_json_docs(json_path: str | Path) -> list[dict]:
    """Load raw docs from a JSON file."""
    path = Path(json_path)
    if not path.exists():
        logger.warning("JSON file not found: %s", path)
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ingest_all_from_disk() -> int:
    """
    Scan raw data directories and ingest all JSON files into ChromaDB.
    Useful for re-building the index from scratch.
    """
    raw_dir = Path(settings.raw_data_dir)
    total = 0

    for json_file in raw_dir.rglob("*.json"):
        docs = load_json_docs(json_file)
        if docs:
            count = ingest_documents(docs)
            total += count
            logger.info("  %s → %d docs ingested", json_file.name, count)

    logger.info("Total ingested: %d documents", total)
    return total


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    total = ingest_all_from_disk()
    print(f"\nTotal documents ingested: {total}")
