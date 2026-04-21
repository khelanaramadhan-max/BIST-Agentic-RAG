"""
Seed script to generate extensive BIST data (RAW, Chroma DB, PDFs)
for BIST 100 companies and Turkish Banks.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Adjust path so we can import from top level
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from ingestion.kap_scraper import fetch_disclosures_for_ticker, save_disclosures
from ingestion.news_fetcher import fetch_news_for_ticker, save_news
from ingestion.pdf_parser import parse_all_pdfs
from ingestion.embedder import ingest_documents

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Massive list from BIST 100 + Banks
BIST_TICKERS = [
    # Banks
    "AKBNK", "GARAN", "ISCTR", "YKBNK", "VAKBN", "HALKB", "ALBRK", "SKBNK", "TSKB",
    # Industrials/Conglomerates/Tech (BIST 30/50 proxies)
    "THYAO", "BIMAS", "KCHOL", "EREGL", "ASELS", "TUPRS", "SAHOL", "SISE", 
    "ENKAI", "PETKM", "FROTO", "PGSUS", "TOASO", "TCELL", "TTKOM", "KRDMD", 
    "SASA", "HEKTS", "KORDS", "MGROS", "SELEC", "DOAS", "TAVHL",
    # Additional BIST Market Stocks
    "AKSA", "ALARK", "ARCLK", "ASTOR", "BRSAN", "CCOLA", "CIMSA", "CWENE", 
    "DOHOL", "ECILC", "EGEEN", "EKGYO", "ENJSA", "EUPWR", "GESAN", "GUBRF", 
    "GWIND", "IPEKE", "ISGYO", "KLRHO", "KLSER", "KONTR", "KOZAA", "KOZAL", 
    "MAVI", "MIATK", "ODAS", "OYAKC", "PENTA", "SMRTG", "SOKM", "TKFEN", 
    "VESBE", "VESTL", "YYLGD"
]

def generate_mock_broker_pdfs():
    """Generate realistic mock PDF broker reports using reportlab if available."""
    if not REPORTLAB_AVAILABLE:
        logger.warning("reportlab not installed. Skipping PDF generation.")
        return

    pdf_dir = Path(settings.raw_data_dir) / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    
    brokers = ["IsYatirim", "YapiKredi", "GarantiBBVA", "AkYatirim", "Tera", "ZiraatYatirim", "Gedik", "VakifYatirim"]
    import random

    for ticker in BIST_TICKERS:
        broker = random.choice(brokers)
        target_price = round(random.uniform(10, 500), 2)
        rating = random.choice(["Buy", "Hold", "Outperform", "Strong Buy", "Overweight"])
        text = f"{ticker} shows resilience in the current macroeconomic environment. Margins have stabilized leading to steady cash flows. {rating} rating initiated with a target price of {target_price} TRY."
        
        filename = f"{ticker}_{broker}_Initiation_{today}.pdf"
        filepath = pdf_dir / filename
        
        c = canvas.Canvas(str(filepath), pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 700, f"{broker} - Equity Research: {ticker}")
        
        c.setFont("Helvetica", 12)
        c.drawString(72, 670, f"Date: {today}")
        
        # Split text into lines just in case
        lines = text.split(". ")
        y = 620
        for line in lines:
            c.drawString(72, y, line + ("." if not line.endswith(".") else ""))
            y -= 20
            
        c.save()
        logger.info(f"Generated mock PDF: {filename}")

def run_seeding():
    logger.info(f"Starting massive BIST data seed for {len(BIST_TICKERS)} tickers...")
    
    # 1. Generate PDFs
    generate_mock_broker_pdfs()
    
    all_docs = []

    # 2. Fetch KAP and News
    for ticker in BIST_TICKERS:
        logger.info(f"--- Fetching {ticker} ---")
        try:
            kap = fetch_disclosures_for_ticker(ticker, limit=5, days_back=90)
            if kap:
                save_disclosures(kap, ticker)
                all_docs.extend(kap)
            
            news = fetch_news_for_ticker(ticker, days_back=30, max_articles=5)
            if news:
                save_news(news, ticker)
                all_docs.extend(news)
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")

    # 3. Parse PDFs
    logger.info("Parsing PDFs...")
    try:
        pdfs = parse_all_pdfs()
        if pdfs:
            all_docs.extend(pdfs)
    except Exception as e:
        logger.error(f"Error parsing PDFs: {e}")

    # 4. Ingest into Chroma
    logger.info(f"Embedding {len(all_docs)} total docs into ChromaDB...")
    try:
        if all_docs:
            ingest_documents(all_docs)
            logger.info("ChromaDB ingestion successful!")
        else:
            logger.warning("No documents found to ingest.")
    except Exception as e:
        logger.error(f"Chroma DB embedding failed: {e}")

if __name__ == "__main__":
    run_seeding()
