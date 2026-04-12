"""
KAP Scraper – fetches public disclosures from kap.org.tr
Treats KAP as the authoritative / ground-truth source.
"""

from __future__ import annotations

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from config.settings import settings

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# KAP internal API endpoints (discovered via browser dev-tools)
KAP_DISCLOSURE_API = "https://www.kap.org.tr/tr/api/memberDisclosureQuery"
KAP_COMPANY_API = "https://www.kap.org.tr/tr/api/memberList"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _sleep():
    time.sleep(settings.kap_request_delay)


def _get(url: str, params: dict | None = None) -> requests.Response | None:
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        _sleep()
        return resp
    except Exception as exc:
        logger.warning("KAP GET failed for %s: %s", url, exc)
        return None


# ─── Company List ─────────────────────────────────────────────────────────────


def fetch_company_list() -> list[dict]:
    """Return a list of BIST companies from KAP's member list."""
    resp = _get(KAP_COMPANY_API)
    if resp is None:
        return _fallback_company_list()
    try:
        data = resp.json()
        companies = []
        for item in data:
            companies.append(
                {
                    "ticker": item.get("memberCode", ""),
                    "name": item.get("memberDesc", ""),
                    "member_id": item.get("memberId", ""),
                }
            )
        return companies
    except Exception as exc:
        logger.error("Error parsing company list: %s", exc)
        return _fallback_company_list()


def _fallback_company_list() -> list[dict]:
    """Hardcoded top BIST companies when scraping fails."""
    return [
        {"ticker": "ASELS", "name": "Aselsan Elektronik Sanayi ve Ticaret A.Ş.", "member_id": ""},
        {"ticker": "AKBNK", "name": "Akbank T.A.Ş.", "member_id": ""},
        {"ticker": "GARAN", "name": "Türkiye Garanti Bankası A.Ş.", "member_id": ""},
        {"ticker": "EREGL", "name": "Ereğli Demir ve Çelik Fabrikaları T.A.Ş.", "member_id": ""},
        {"ticker": "KCHOL", "name": "Koç Holding A.Ş.", "member_id": ""},
        {"ticker": "THYAO", "name": "Türk Hava Yolları A.O.", "member_id": ""},
        {"ticker": "BIMAS", "name": "BİM Birleşik Mağazalar A.Ş.", "member_id": ""},
        {"ticker": "TUPRS", "name": "Tüpraş-Türkiye Petrol Rafinerileri A.Ş.", "member_id": ""},
        {"ticker": "SISE", "name": "Türkiye Şişe ve Cam Fabrikaları A.Ş.", "member_id": ""},
        {"ticker": "PGSUS", "name": "Pegasus Hava Taşımacılığı A.Ş.", "member_id": ""},
    ]


# ─── Disclosure Fetching ──────────────────────────────────────────────────────


def fetch_disclosures_for_ticker(
    ticker: str, limit: int = 20, days_back: int = 180
) -> list[dict[str, Any]]:
    """
    Fetch recent KAP disclosures for a single ticker.
    Falls back to HTML scraping if JSON API fails.
    Returns a list of document dicts with standardised metadata.
    """
    logger.info("Fetching KAP disclosures for %s (last %d days)...", ticker, days_back)

    # ── Try JSON endpoint first ──────────────────────────────
    docs = _fetch_via_api(ticker, limit, days_back)
    if docs:
        logger.info("  Found %d disclosures via API for %s", len(docs), ticker)
        return docs

    # ── Fallback: HTML scraping ──────────────────────────────
    docs = _fetch_via_html(ticker, limit)
    logger.info("  Found %d disclosures via HTML scrape for %s", len(docs), ticker)
    return docs


def _fetch_via_api(ticker: str, limit: int, days_back: int) -> list[dict]:
    """Hit KAP's internal JSON API."""
    url = f"https://www.kap.org.tr/tr/api/memberList"
    # Search by memberCode (ticker)
    company_resp = _get(KAP_COMPANY_API)
    if company_resp is None:
        return []
    try:
        companies = company_resp.json()
        member_id = next(
            (c.get("memberId") for c in companies if c.get("memberCode") == ticker),
            None,
        )
    except Exception:
        return []

    if not member_id:
        return []

    # Fetch disclosures for this member
    disc_url = f"https://www.kap.org.tr/tr/api/memberDisclosureQuery/{member_id}"
    resp = _get(disc_url)
    if resp is None:
        return []

    try:
        items = resp.json()
        docs = []
        for item in items[:limit]:
            docs.append(
                _normalise_kap_item(item, ticker)
            )
        return docs
    except Exception:
        return []


def _fetch_via_html(ticker: str, limit: int) -> list[dict]:
    """Scrape KAP member disclosure page HTML."""
    url = f"{settings.kap_base_url}/tr/sirket-bilgileri/ozet/{ticker}/"
    resp = _get(url)
    if resp is None:
        return _generate_sample_disclosures(ticker)

    soup = BeautifulSoup(resp.text, "lxml")
    docs = []

    # Try to find disclosure rows in common KAP table patterns
    rows = soup.select("table tr, .w-clearfix.w-inline-block")
    for row in rows[:limit]:
        text = row.get_text(separator=" ", strip=True)
        if len(text) < 30:
            continue
        docs.append(
            {
                "ticker": ticker,
                "source_type": "kap_disclosure",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "institution": "KAP",
                "title": text[:120],
                "content": text,
                "url": url,
                "disclosure_type": "material_event",
            }
        )

    if not docs:
        docs = _generate_sample_disclosures(ticker)
    return docs


def _normalise_kap_item(item: dict, ticker: str) -> dict:
    """Map a raw KAP API item to our standard schema."""
    return {
        "ticker": ticker,
        "source_type": "kap_disclosure",
        "date": item.get("disclosureDate", item.get("date", datetime.now().strftime("%Y-%m-%d"))),
        "institution": "KAP",
        "title": item.get("title", item.get("subject", "KAP Disclosure")),
        "content": item.get("content", item.get("description", "")),
        "url": item.get("url", f"{settings.kap_base_url}/tr/"),
        "disclosure_type": item.get("disclosureType", "material_event"),
        "summary_flag": item.get("summaryFlag", False),
    }


def _generate_sample_disclosures(ticker: str) -> list[dict]:
    """
    Generate realistic sample disclosures for demo purposes.
    This is used when live scraping is not possible.
    Clearly marked as SAMPLE data.
    """
    now = datetime.now()
    samples = [
        {
            "ticker": ticker,
            "source_type": "kap_disclosure",
            "date": now.strftime("%Y-%m-01"),
            "institution": "KAP [SAMPLE DATA]",
            "title": f"{ticker} — Özel Durum Açıklaması: Yönetim Kurulu Kararı",
            "content": (
                f"{ticker} Yönetim Kurulu, şirketin 2024 yılı ikinci çeyreğine ilişkin "
                f"finansal sonuçlarını değerlendirmiş ve temettü dağıtımı konusunda karar almıştır. "
                f"Detaylı bilgi KAP sistemi üzerinden kamuoyuyla paylaşılmıştır. "
                f"[SAMPLE – canlı KAP verisi çekilemedi]"
            ),
            "url": f"https://www.kap.org.tr/tr/sirket-bilgileri/ozet/{ticker}/",
            "disclosure_type": "board_decision",
        },
        {
            "ticker": ticker,
            "source_type": "kap_disclosure",
            "date": now.strftime("%Y-%m-15"),
            "institution": "KAP [SAMPLE DATA]",
            "title": f"{ticker} — Finansal Rapor Özeti (Q2 2024)",
            "content": (
                f"{ticker} A.Ş., 2024 yılı ikinci çeyrek finansal tablolarını KAP'a bildirmiştir. "
                f"Net satışlar bir önceki yılın aynı dönemine kıyasla artış göstermiştir. "
                f"FAVÖK marjı sektör ortalamasıyla uyumludur. "
                f"[SAMPLE – canlı KAP verisi çekilemedi]"
            ),
            "url": f"https://www.kap.org.tr/tr/sirket-bilgileri/ozet/{ticker}/",
            "disclosure_type": "financial_report_summary",
        },
        {
            "ticker": ticker,
            "source_type": "kap_disclosure",
            "date": now.strftime("%Y-%m-20"),
            "institution": "KAP [SAMPLE DATA]",
            "title": f"{ticker} — Bağımsız Denetçi Görüşü",
            "content": (
                f"Bağımsız denetim firması, {ticker} A.Ş.'nin finansal tablolarını incelemiş "
                f"ve olumlu görüş bildirmiştir. Önemli bir ihtirazi kayıt bulunmamaktadır. "
                f"[SAMPLE – canlı KAP verisi çekilemedi]"
            ),
            "url": f"https://www.kap.org.tr/tr/sirket-bilgileri/ozet/{ticker}/",
            "disclosure_type": "audit_report",
        },
    ]
    return samples


# ─── Batch Save ───────────────────────────────────────────────────────────────


def save_disclosures(docs: list[dict], ticker: str) -> Path:
    """Persist fetched disclosures to disk as JSON."""
    out_dir = Path(settings.raw_data_dir) / "kap"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}_disclosures.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d disclosures → %s", len(docs), out_path)
    return out_path


# ─── CLI ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "ASELS"
    logging.basicConfig(level=logging.INFO)
    docs = fetch_disclosures_for_ticker(ticker)
    save_disclosures(docs, ticker)
    print(f"\nFetched {len(docs)} disclosures for {ticker}")
    for d in docs[:3]:
        print(f"  [{d['date']}] {d['title'][:80]}...")
