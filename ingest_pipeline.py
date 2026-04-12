"""
One-shot ingestion script: fetches KAP + news for a list of tickers
and embeds everything into ChromaDB.
Run: python ingest_pipeline.py [TICKER1 TICKER2 ...]
"""

from __future__ import annotations

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("ingest_pipeline")

DEFAULT_TICKERS = ["ASELS", "GARAN", "AKBNK", "THYAO", "BIMAS", "EREGL", "KCHOL"]


def run_ingestion(tickers: list[str]) -> None:
    from ingestion.kap_scraper import fetch_disclosures_for_ticker, save_disclosures
    from ingestion.news_fetcher import fetch_news_for_ticker, save_news
    from ingestion.pdf_parser import parse_all_pdfs
    from ingestion.embedder import ingest_documents, ingest_all_from_disk

    all_docs = []

    for ticker in tickers:
        logger.info("══════════════════════════════════════")
        logger.info("Processing ticker: %s", ticker)

        # ── KAP ──────────────────────────────
        logger.info("[1/2] Fetching KAP disclosures...")
        kap_docs = fetch_disclosures_for_ticker(ticker, limit=20, days_back=180)
        if kap_docs:
            save_disclosures(kap_docs, ticker)
            all_docs.extend(kap_docs)
            logger.info("  KAP: %d disclosures collected.", len(kap_docs))

        # ── News ─────────────────────────────
        logger.info("[2/2] Fetching financial news...")
        news_docs = fetch_news_for_ticker(ticker, days_back=90, max_articles=10)
        if news_docs:
            save_news(news_docs, ticker)
            all_docs.extend(news_docs)
            logger.info("  News: %d articles collected.", len(news_docs))

    # ── PDFs (optional) ──────────────────────
    logger.info("══════════════════════════════════════")
    logger.info("[3/3] Parsing brokerage PDFs (if any)...")
    pdf_docs = parse_all_pdfs()
    if pdf_docs:
        all_docs.extend(pdf_docs)
        logger.info("  PDFs: %d chunks parsed.", len(pdf_docs))

    # ── Embed all into ChromaDB ───────────────
    logger.info("══════════════════════════════════════")
    logger.info("Embedding %d total documents into ChromaDB...", len(all_docs))
    total_ingested = ingest_documents(all_docs)
    logger.info("✅ Total ingested: %d documents.", total_ingested)

    # Show collection stats
    from vectordb.chroma_store import collection_stats
    stats = collection_stats()
    logger.info("ChromaDB collection sizes: %s", stats)


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS
    logger.info("Starting ingestion pipeline for tickers: %s", tickers)
    run_ingestion(tickers)
    logger.info("Done! You can now run the API or Streamlit UI.")
