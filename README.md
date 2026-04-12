# BIST Equity Intelligence Agent 📊

> **Agentic RAG system for Turkish equity market intelligence.**  
> Combines KAP (Kamu Aydınlatma Platformu) disclosures, financial news, and brokerage research reports to deliver evidence-based, source-cited, time-aware market analysis.

> 🚀 **Turkish Market Intelligence:** This system is powered by AI for deep narrative analysis and market intelligence.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
FastAPI  ←→  Streamlit UI  ←→  HTML Demo
    │
    ▼
LangGraph Agentic Loop
  1. route_query   → decides KAP / News / Brokerage
  2. retrieve      → ChromaDB similarity search
  3. grade         → is context sufficient?
  4. rewrite       → query reformulation (if needed)
  5. answer        → Groq Llama 3.3 70B generates cited answer
  6. guardrail     → checks for investment advice, injects disclaimer
    │
    ▼
Response (cited, time-aware, non-advisory)
```

## 📁 Project Structure

```
BIST-Agentic-RAG/
├── .env.example            # Copy to .env and fill in keys
├── requirements.txt        # All Python dependencies
├── docker-compose.yml      # API + UI containers
├── ingest_pipeline.py      # One-shot data ingestion
│
├── config/settings.py      # Pydantic settings
├── ingestion/
│   ├── kap_scraper.py      # KAP.org.tr HTML/API scraper
│   ├── news_fetcher.py     # RSS financial news (Bloomberg HT, Dünya, etc.)
│   ├── pdf_parser.py       # Brokerage PDF extraction (PyMuPDF)
│   └── embedder.py         # Chunk + embed → ChromaDB
│
├── vectordb/chroma_store.py  # ChromaDB retriever factory
│
├── agent/
│   ├── graph.py            # LangGraph state machine
│   ├── tools.py            # Retriever tools per source type
│   └── prompts.py          # All prompt templates
│
├── guardrails/checker.py   # Ethical guardrails + disclaimer
├── evaluation/
│   ├── questions.py        # 12 BIST-specific eval questions
│   └── evaluator.py        # Faithfulness, relevancy, coverage metrics
│
├── api/main.py             # FastAPI REST API
└── ui/
    ├── app.py              # Streamlit UI
    └── index.html          # Standalone HTML demo
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/BIST-Agentic-RAG.git
cd BIST-Agentic-RAG

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your keys
```

Your `.env` must contain:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at https://console.groq.com/

### 3. Ingest Data

```bash
# Ingest KAP disclosures + news for default tickers
python ingest_pipeline.py

# Or for specific tickers
python ingest_pipeline.py ASELS GARAN THYAO
```

To add brokerage PDFs, place them in `data/raw/pdfs/` then run:
```bash
python -c "from ingestion.pdf_parser import parse_all_pdfs; from ingestion.embedder import ingest_documents; ingest_documents(parse_all_pdfs())"
```

### 4. Run the API

```bash
python -m uvicorn api.main:app --reload --port 8000
```

API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### 5. Run the UI

**Option A — Streamlit (recommended):**
```bash
streamlit run ui/app.py
```
→ Open http://localhost:8501

**Option B — HTML demo:**
Open `ui/index.html` in your browser while the API is running.

---

## 🐳 Docker

```bash
# Build and run both API + Streamlit
docker-compose up --build

# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## 💬 Sample Queries

| Category | Question |
|----------|----------|
| KAP-Centric | `ASELS son 6 ayda hangi tür KAP açıklamaları yaptı?` |
| Brokerage | `GARAN için son araştırma raporlarında hangi ortak temalar öne çıkıyor?` |
| Consistency | `BIMAS hakkındaki son haberler resmi KAP açıklamalarıyla tutarlı mı?` |
| Narrative | `THYAO etrafındaki anlatı son 6 ayda nasıl değişti?` |
| Sector | `Türk bankacılık sektörüne ilişkin son açıklamalar ne söylüyor?` |

## 🔌 API Reference

### POST `/query`
```json
{
  "question": "ASELS son KAP açıklamalarını özetle",
  "ticker": "ASELS",
  "chat_history": []
}
```

### POST `/ingest`
```json
{
  "ticker": "GARAN",
  "include_kap": true,
  "include_news": true,
  "days_back": 90
}
```

### GET `/stats`
Returns document counts per ChromaDB collection.

### POST `/evaluate`
Runs the BIST-specific evaluation pipeline.

---

## 📊 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM | Groq + Llama 3.3 70B | Ultra-low latency inference |
| Framework | LangGraph + LangChain | Agentic loop state machine |
| Vector DB | ChromaDB (local) | Document storage & retrieval |
| Embeddings | paraphrase-multilingual-MiniLM | Turkish language support |
| Data — KAP | requests + BeautifulSoup | HTML scraping |
| Data — News | feedparser + RSS | Turkish financial news |
| Data — PDF | PyMuPDF + pdfplumber | Brokerage report parsing |
| Guardrails | Custom checker | Quality assurance |
| Evaluation | Custom RAGAS-style | Faithfulness, relevancy |
| API | FastAPI | REST interface |
| UI | Streamlit + HTML | Demo frontends |

## 📋 Evaluation Metrics

The system evaluates against 12 BIST-specific questions across 4 categories:

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Are claims grounded in retrieved context? |
| **Answer Relevancy** | Does the answer address the question? |
| **Source Coverage** | Were the right sources (KAP/news/brokerage) used? |
| **Disclaimer Present** | Is the non-advisory disclaimer included? |
| **Non-Advisory** | Is the answer free of investment signals? |

Run evaluation:
```bash
python evaluation/evaluator.py 5
```

## 🧱 Metadata Schema (ChromaDB)

All documents stored with this mandatory metadata:

```python
{
    "ticker": "ASELS",           # BIST ticker code
    "source_type": "kap_disclosure",  # kap_disclosure | news | brokerage_report
    "date": "2024-01-15",        # Publication date
    "institution": "KAP",        # Source institution
    "title": "...",              # Document title
    "url": "https://...",        # Source URL
}
```

## ⚖️ Ethical Constraints

The system enforces these hard rules at every layer:

1. **No investment advice** — prompts explicitly forbid buy/sell signals
2. **No price predictions** — pattern matching blocks price targets
3. **No return forecasts** — LLM and regex guardrails combined
4. **Mandatory disclaimer** — injected into every response

---

## 📚 Assignment Coverage

| Requirement | Implementation |
|-------------|---------------|
| KAP disclosures (HTML source) | `ingestion/kap_scraper.py` |
| Brokerage PDFs | `ingestion/pdf_parser.py` |
| Financial news | `ingestion/news_fetcher.py` |
| Agentic loop | `agent/graph.py` (LangGraph) |
| Source routing | `agent/graph.py` → `route_query` |
| Cross-source verification | `agent/prompts.py` → `CONSISTENCY_PROMPT` |
| Iterative retrieval | `agent/graph.py` → `grade_context` → `rewrite_query` |
| Memory | LangGraph state + chat_history |
| Guardrails | `guardrails/checker.py` |
| Evaluation (10+ questions) | `evaluation/questions.py` (12 questions) |
| Docker | `Dockerfile` + `docker-compose.yml` |
| Vector DB metadata schema | ticker, source_type, date, institution |
