"""
News Fetcher – pulls Turkish financial/market news from RSS feeds and free endpoints.
Sources: Reuters TR, Bloomberg HT, Investing.com TR, Dünya Gazetesi.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import feedparser
import requests
from bs4 import BeautifulSoup

from config.settings import settings

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
}

# Free RSS feeds for Turkish financial news
RSS_FEEDS = [
    {
        "name": "Bloomberg HT Piyasalar",
        "url": "https://www.bloomberght.com/rss",
        "institution": "Bloomberg HT",
    },
    {
        "name": "Dünya Gazetesi Ekonomi",
        "url": "https://www.dunya.com/rss/ekonomi.xml",
        "institution": "Dünya Gazetesi",
    },
    {
        "name": "Ekonomim",
        "url": "https://www.ekonomim.com/rss.xml",
        "institution": "Ekonomim",
    },
    {
        "name": "Finans Gündem",
        "url": "https://www.finansgundem.com/rss.xml",
        "institution": "Finans Gündem",
    },
    {
        "name": "Para Analiz",
        "url": "https://www.paraanaliz.com/rss",
        "institution": "Para Analiz",
    },
]


# ─── RSS Fetcher ─────────────────────────────────────────────────────────────


def fetch_news_for_ticker(
    ticker: str,
    company_name: str = "",
    days_back: int = 30,
    max_articles: int = 15,
) -> list[dict[str, Any]]:
    """
    Fetch recent news articles mentioning the ticker or company name.
    Returns normalised document dicts.
    """
    logger.info("Fetching news for %s ...", ticker)
    
    # If general market query, don't filter aggressively
    is_general = ticker.upper() in ("BIST", "MACRO", "BIST 100", "GLOBAL")
    search_terms = [] if is_general else [ticker.upper()]
    
    if company_name and not is_general:
        # Use first meaningful word from company name
        first_word = company_name.split()[0] if company_name else ""
        if first_word and first_word not in search_terms:
            search_terms.append(first_word)

    cutoff = datetime.now() - timedelta(days=days_back)
    articles: list[dict] = []

    for feed_info in RSS_FEEDS:
        feed_articles = _parse_rss_feed(feed_info, search_terms, cutoff)
        articles.extend(feed_articles)
        if len(articles) >= max_articles:
            break

    # De-duplicate by title
    seen_titles: set[str] = set()
    unique = []
    for art in articles:
        key = art["title"][:60].lower()
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(art)

    # Tag everything
    for art in unique:
        art["ticker"] = ticker

    if not unique:
        logger.info("No live news found for %s – using samples.", ticker)
        unique = _generate_sample_news(ticker, company_name)

    logger.info("  Collected %d news articles for %s", len(unique), ticker)
    return unique[:max_articles]


def _parse_rss_feed(
    feed_info: dict, search_terms: list[str], cutoff: datetime
) -> list[dict]:
    """Parse a single RSS feed and filter by search terms."""
    try:
        feed = feedparser.parse(feed_info["url"])
        results = []
        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            text = (title + " " + summary).upper()

            if search_terms and not any(term.upper() in text for term in search_terms):
                continue

            pub_date = _parse_date(entry)
            if pub_date < cutoff:
                continue

            results.append(
                {
                    "source_type": "news",
                    "institution": feed_info["institution"],
                    "date": pub_date.strftime("%Y-%m-%d"),
                    "title": title.strip(),
                    "content": BeautifulSoup(summary, "html.parser").get_text(
                        separator=" ", strip=True
                    ),
                    "url": entry.get("link", ""),
                }
            )
        return results
    except Exception as exc:
        logger.warning("RSS parse error for %s: %s", feed_info["url"], exc)
        return []


def _parse_date(entry) -> datetime:
    """Best-effort date parsing from an RSS entry."""
    for attr in ("published_parsed", "updated_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return datetime(*val[:6])
            except Exception:
                pass
    return datetime.now()


# ─── Investing.com TR Search (HTML) ──────────────────────────────────────────


def fetch_investing_news(ticker: str, max_articles: int = 5) -> list[dict]:
    """Scrape Investing.com Turkey search for additional articles."""
    try:
        url = f"https://tr.investing.com/search/?q={ticker}&tab=news"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")
        articles = []
        for item in soup.select(".js-article-item")[:max_articles]:
            title_el = item.select_one(".title")
            date_el = item.select_one(".date")
            if not title_el:
                continue
            articles.append(
                {
                    "ticker": ticker,
                    "source_type": "news",
                    "institution": "Investing.com TR",
                    "date": date_el.get_text(strip=True) if date_el else datetime.now().strftime("%Y-%m-%d"),
                    "title": title_el.get_text(strip=True),
                    "content": title_el.get_text(strip=True),
                    "url": "https://tr.investing.com" + (item.get("href") or ""),
                }
            )
        return articles
    except Exception as exc:
        logger.warning("Investing.com scrape error: %s", exc)
        return []


# ─── Sample News Generator ───────────────────────────────────────────────────


def _generate_sample_news(ticker: str, company_name: str = "") -> list[dict]:
    """
    Realistic sample news articles used when live feeds return nothing.
    Clearly labelled as SAMPLE data.
    """
    name = company_name or ticker
    now = datetime.now()

    return [
        {
            "ticker": ticker,
            "source_type": "news",
            "institution": "Bloomberg HT [SAMPLE]",
            "date": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
            "title": f"{ticker}: Analistler güçlü çeyrek beklentisi koruyor",
            "content": f"{name}, son dönemde yurt içi tüketim talebindeki artış ve ihracat gelirlerindeki iyileşme sayesinde olumlu finansal görünümünü sürdürmektedir.",
            "url": "https://www.bloomberght.com/",
        },
        {
            "ticker": ticker,
            "source_type": "news",
            "institution": "Dünya Gazetesi [SAMPLE]",
            "date": (now - timedelta(days=2)).strftime("%Y-%m-%d"),
            "title": f"{ticker} sektöründe dönüşüm hızlanıyor",
            "content": f"Türkiye'nin önde gelen şirketlerinden {name}, dijital dönüşüm yatırımlarına hız verdiğini açıkladı.",
            "url": "https://www.dunya.com/",
        },
        {
            "ticker": ticker,
            "source_type": "news",
            "institution": "Para Analiz [SAMPLE]",
            "date": (now - timedelta(days=3)).strftime("%Y-%m-%d"),
            "title": f"BIST'te {ticker} izleme listesinde",
            "content": f"Borsa İstanbul'da {ticker} hissesi, son haftalarda yoğun işlem hacmiyle dikkat çekmektedir.",
            "url": "https://www.paraanaliz.com/",
        },
        {
            "ticker": "MACRO",
            "source_type": "news",
            "institution": "Ekonomim [SAMPLE]",
            "date": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
            "title": "TCMB faiz kararı öncesi piyasalarda beklenti anketi sonuçlandı",
            "content": "Merkez Bankası'nın faiz kararı piyasalar tarafından dikkatle bekleniyor.",
            "url": "https://www.ekonomim.com/",
        },
        {
            "ticker": "GLOBAL",
            "source_type": "news",
            "institution": "Bloomberg HT [SAMPLE]",
            "date": (now - timedelta(hours=5)).strftime("%Y-%m-%d"),
            "title": "FED tutanakları enflasyon endişelerinin sürdüğüne işaret ediyor",
            "content": "FED yetkilileri enflasyonla mücadelede henüz erken zafer ilan edilmemesi gerektiğini düşünüyor.",
            "url": "https://www.bloomberght.com/",
        },
        {
            "ticker": ticker,
            "source_type": "news",
            "institution": "Finans Gündem [SAMPLE]",
            "date": (now - timedelta(hours=14)).strftime("%Y-%m-%d"),
            "title": f"{ticker} hisselerinde yabancı ilgisi artıyor",
            "content": "Yabancı yatırımcıların Borsa İstanbul'a yönelik ilgisi son raporlara da yansıdı.",
            "url": "https://www.finansgundem.com/",
        },
        {
            "ticker": "BTC-USD",
            "source_type": "news",
            "institution": "CoinDesk TR [SAMPLE]",
            "date": (now - timedelta(hours=2)).strftime("%Y-%m-%d"),
            "title": "Bitcoin dalgalı seyrine devam ederken kripto ETF'lere girişler artıyor",
            "content": "Kripto piyasalarında Bitcoin ETF'lerine yönelik kurumsal ilginin artmasıyla fiyatlarda dalgalı seyir devam ediyor.",
            "url": "https://tr.investing.com/crypto/bitcoin/news",
        },
        {
            "ticker": ticker,
            "source_type": "news",
            "institution": "Dünya Gazetesi [SAMPLE]",
            "date": (now - timedelta(hours=6)).strftime("%Y-%m-%d"),
            "title": f"{ticker} için yeni hedef fiyat açıklandı",
            "content": "Aracı kurumlar son finansallar sonrası şirket için hedef fiyatı yukarı yönlü revize etti.",
            "url": "https://www.dunya.com/",
        },
        {
            "ticker": "MACRO",
            "source_type": "news",
            "institution": "Bloomberg HT [SAMPLE]",
            "date": (now - timedelta(days=4)).strftime("%Y-%m-%d"),
            "title": "İhracat rakamlarında rekor beklentisi yetkililerce doğrulandı",
            "content": "Yılın son çeyreğinde ihracat artışının sürmesi bekleniyor.",
            "url": "https://www.bloomberght.com/",
        },
        {
            "ticker": "GLOBAL",
            "source_type": "news",
            "institution": "Para Analiz [SAMPLE]",
            "date": (now - timedelta(days=2)).strftime("%Y-%m-%d"),
            "title": "Altın fiyatlarında jeopolitik risklerin etkisiyle tarihi zirve test edildi",
            "content": "Uluslararası piyasalarda ons altın, küresel gerilimlerin artmasıyla yönünü yukarı çevirdi.",
            "url": "https://www.paraanaliz.com/",
        }
    ]


# ─── Save ────────────────────────────────────────────────────────────────────


def save_news(docs: list[dict], ticker: str) -> Path:
    out_dir = Path(settings.raw_data_dir) / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}_news.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d news articles → %s", len(docs), out_path)
    return out_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    ticker = sys.argv[1] if len(sys.argv) > 1 else "GARAN"
    docs = fetch_news_for_ticker(ticker)
    save_news(docs, ticker)
    print(f"\n{len(docs)} articles for {ticker}")
    for d in docs[:3]:
        print(f"  [{d['date']}] [{d['institution']}] {d['title'][:70]}...")
