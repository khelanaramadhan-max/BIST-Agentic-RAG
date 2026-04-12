"""
PDF Parser – extracts text from brokerage equity research reports.
Supports PyMuPDF (primary) and pdfplumber (fallback).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


# ─── Extraction ──────────────────────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract full text from a PDF using PyMuPDF, fallback to pdfplumber."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Primary: PyMuPDF
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        text = "\n".join(pages)
        if len(text.strip()) > 100:
            return text
    except ImportError:
        logger.warning("PyMuPDF not installed – trying pdfplumber.")
    except Exception as exc:
        logger.warning("PyMuPDF error: %s", exc)

    # Fallback: pdfplumber
    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except Exception as exc:
        logger.error("pdfplumber error: %s", exc)
        return ""


# ─── Metadata Extraction ─────────────────────────────────────────────────────


def extract_metadata_from_text(text: str, filename: str) -> dict[str, Any]:
    """
    Heuristically extract ticker, date, and institution from PDF text.
    """
    meta: dict[str, Any] = {
        "source_type": "brokerage_report",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "institution": "Unknown Brokerage",
        "ticker": "",
        "filename": filename,
    }

    # ── Ticker detection ─────────────────────────────────────
    ticker_patterns = [
        r"\b([A-Z]{3,5}\.IS)\b",          # Bloomberg format: GARAN.IS
        r"Hisse Kodu[:\s]+([A-Z]{3,5})",  # Turkish label
        r"Ticker[:\s]+([A-Z]{3,5})",
        r"KAP Kodu[:\s]+([A-Z]{3,5})",
    ]
    for pattern in ticker_patterns:
        m = re.search(pattern, text[:3000])
        if m:
            meta["ticker"] = m.group(1).replace(".IS", "")
            break

    # Fallback: guess from filename (e.g., GARAN_report.pdf)
    if not meta["ticker"]:
        fn_match = re.match(r"([A-Z]{3,5})", filename.upper())
        if fn_match:
            meta["ticker"] = fn_match.group(1)

    # ── Institution detection ─────────────────────────────────
    known_brokerages = [
        "Garanti BBVA", "İş Yatırım", "Yapı Kredi Yatırım",
        "Ak Yatırım", "Gedik Yatırım", "Deniz Yatırım",
        "TEB Yatırım", "Ata Yatırım", "Halk Yatırım",
        "Şeker Yatırım", "Logo Yatırım", "HSBC",
        "JPMorgan", "Goldman Sachs", "Morgan Stanley",
    ]
    for brokerage in known_brokerages:
        if brokerage.lower() in text[:2000].lower():
            meta["institution"] = brokerage
            break

    # ── Date detection ────────────────────────────────────────
    date_patterns = [
        r"(\d{2}[./]\d{2}[./]\d{4})",
        r"(\d{4}-\d{2}-\d{2})",
        r"(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})",
    ]
    for pattern in date_patterns:
        m = re.search(pattern, text[:3000])
        if m:
            raw = m.group(1)
            try:
                for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"):
                    try:
                        meta["date"] = datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
            break

    return meta


# ─── Chunking ────────────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks preserving paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # If para itself is too long, hard-split it
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i : i + chunk_size])
                current = para[-(overlap):]
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


# ─── High-level: parse + chunk + save ────────────────────────────────────────


def parse_pdf_to_documents(pdf_path: str | Path) -> list[dict[str, Any]]:
    """
    End-to-end: extract text, detect metadata, chunk, return list of doc dicts.
    """
    pdf_path = Path(pdf_path)
    logger.info("Parsing PDF: %s", pdf_path.name)

    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        logger.warning("Empty text extracted from %s", pdf_path.name)
        return []

    meta = extract_metadata_from_text(raw_text, pdf_path.name)
    chunks = chunk_text(
        raw_text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    docs = []
    for i, chunk in enumerate(chunks):
        doc = dict(meta)  # copy
        doc["content"] = chunk
        doc["chunk_index"] = i
        doc["title"] = f"{meta['institution']} — {meta['ticker']} Report (chunk {i+1})"
        docs.append(doc)

    logger.info(
        "  %s → %d chunks (ticker=%s, institution=%s, date=%s)",
        pdf_path.name, len(docs), meta["ticker"], meta["institution"], meta["date"],
    )
    return docs


def parse_all_pdfs(pdf_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Parse all PDFs in the pdf_dir and return a flat list of doc dicts."""
    pdf_dir = Path(pdf_dir or settings.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found in %s. Add research PDFs to ingest.", pdf_dir)
        return _generate_sample_brokerage_docs()

    all_docs = []
    for pdf in pdfs:
        docs = parse_pdf_to_documents(pdf)
        all_docs.extend(docs)
    return all_docs


def _generate_sample_brokerage_docs() -> list[dict]:
    """Sample brokerage report documents for demo purposes when no PDFs are present."""
    return [
        {
            "ticker": "ASELS",
            "source_type": "brokerage_report",
            "date": datetime.now().strftime("%Y-%m-01"),
            "institution": "İş Yatırım [SAMPLE]",
            "title": "İş Yatırım — ASELS Araştırma Raporu (chunk 1) [SAMPLE]",
            "content": (
                "ASELSAN (ASELS) savunma sektöründeki güçlü sipariş portföyü ve ihracat "
                "gelirlerindeki artış sayesinde olumlu görünümünü korumaktadır. "
                "Yurt içi savunma harcamalarındaki büyüme şirket gelirlerine destek vermektedir. "
                "Operasyonel marjlar beklentilerin üzerinde seyretmekte olup maliyet yönetimi "
                "öne çıkan bir güçlü yön olarak değerlendirilmektedir. "
                "[SAMPLE – gerçek araştırma raporu PDF'i data/raw/pdfs/ klasörüne ekleyin]"
            ),
            "chunk_index": 0,
            "filename": "sample_asels_report.pdf",
        },
        {
            "ticker": "GARAN",
            "source_type": "brokerage_report",
            "date": datetime.now().strftime("%Y-%m-01"),
            "institution": "Ak Yatırım [SAMPLE]",
            "title": "Ak Yatırım — GARAN Banka Araştırma Raporu (chunk 1) [SAMPLE]",
            "content": (
                "Garanti BBVA (GARAN), güçlü kredi büyümesi ve net faiz marjındaki genişleme "
                "ile bankacılık sektörünün üzerinde performans göstermeye devam etmektedir. "
                "Sermaye yeterliliği rasyoları yasal sınırların belirgin biçimde üzerindedir. "
                "Dijital bankacılık kullanıcı sayısındaki artış gelir çeşitlendirmesine katkı "
                "sağlamaktadır. "
                "[SAMPLE – gerçek araştırma raporu PDF'i data/raw/pdfs/ klasörüne ekleyin]"
            ),
            "chunk_index": 0,
            "filename": "sample_garan_report.pdf",
        },
    ]


def save_brokerage_docs(docs: list[dict], out_name: str = "brokerage") -> Path:
    out_dir = Path(settings.raw_data_dir) / "brokerage"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}_docs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d brokerage doc chunks → %s", len(docs), out_path)
    return out_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else settings.pdf_dir
    docs = parse_all_pdfs(path)
    save_brokerage_docs(docs)
    print(f"\nParsed {len(docs)} chunks from PDFs.")
