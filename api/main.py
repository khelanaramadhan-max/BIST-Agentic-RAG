"""
FastAPI REST API for the BIST Agentic RAG system.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from guardrails.checker import check_question_safety

logger = logging.getLogger(__name__)

# ─── Supabase Client ─────────────────────────────────────────────────────────

supabase_client = None
if settings.supabase_url and settings.supabase_key:
    try:
        from supabase import create_client, Client
        supabase_client: Client = create_client(settings.supabase_url, settings.supabase_key)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not initialize Supabase client: {e}")

# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="BIST Equity Intelligence Agent",
    description=(
        "Agentic RAG system for Turkish equity market intelligence. "
        "Sources: KAP disclosures, financial news, brokerage research reports. "
        "⚠️ This system does NOT provide investment advice."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000, description="User question")
    ticker: str = Field("", max_length=10, description="Optional BIST ticker (e.g. ASELS)")
    chat_history: list[dict] = Field(default_factory=list, description="Previous messages")


class QueryResponse(BaseModel):
    answer: str
    sources_used: list[str]
    iteration_count: int
    answer_type: str
    routing_reasoning: str
    grader_confidence: float
    disclaimer: str = (
        "⚠️ This system does not provide investment advice."
    )


class IngestRequest(BaseModel):
    ticker: str = Field(..., min_length=2, max_length=10)
    include_kap: bool = True
    include_news: bool = True
    include_pdfs: bool = False
    days_back: int = Field(90, ge=1, le=365)


class IngestResponse(BaseModel):
    ticker: str
    documents_ingested: int
    kap_count: int
    news_count: int
    brokerage_count: int
    message: str


class StatsResponse(BaseModel):
    collections: dict[str, int]
    status: str


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/", summary="Health check")
def root():
    return {
        "status": "running",
        "system": "BIST Equity Intelligence Agent",
        "version": "1.0.0",
        "disclaimer": "This system does not provide investment advice.",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse, summary="Vector DB collection statistics")
def stats():
    try:
        from vectordb.chroma_store import collection_stats
        return StatsResponse(collections=collection_stats(), status="ok")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse, summary="Ask the BIST Intelligence Agent")
def query(req: QueryRequest):
    """
    Submit a question about a BIST-listed company.
    The agent will:
    1. Route the query to relevant data sources
    2. Retrieve evidence from KAP, news, or brokerage reports
    3. Grade the context and re-retrieve if needed
    4. Generate a cited, time-aware, non-advisory answer
    """
    # Safety pre-check
    is_safe, redirect = check_question_safety(req.question)
    if not is_safe:
        return QueryResponse(
            answer=redirect,
            sources_used=[],
            iteration_count=0,
            answer_type="redirect",
            routing_reasoning="Question blocked by guardrails",
            grader_confidence=0.0,
        )

    # Lazy import to avoid import at startup
    from agent.graph import run_agent
    from langchain_core.messages import HumanMessage, AIMessage

    # Convert chat history
    history = []
    for msg in req.chat_history:
        if msg.get("role") == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            history.append(AIMessage(content=msg["content"]))

    try:
        result = run_agent(
            question=req.question,
            ticker=req.ticker,
            chat_history=history,
        )
        
        # Log to Supabase if configured 
        if supabase_client is not None:
            try:
                supabase_client.table("chat_logs").insert({
                    "question": req.question,
                    "ticker": req.ticker,
                    "answer": result.get("answer", ""),
                    "sources_used": result.get("sources_used", []),
                    "iteration_count": result.get("iteration_count", 0),
                    "grader_confidence": result.get("grader_confidence", 0.0)
                }).execute()
            except Exception as supabase_err:
                logger.warning(f"Failed to log chat to Supabase: {supabase_err}")
                
        return QueryResponse(**result)
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")


@app.post("/ingest", response_model=IngestResponse, summary="Ingest data for a ticker")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Fetch and ingest KAP disclosures and news for a ticker into the vector DB.
    PDF ingestion requires manual file placement in data/raw/pdfs/.
    """
    from ingestion.kap_scraper import fetch_disclosures_for_ticker, save_disclosures
    from ingestion.news_fetcher import fetch_news_for_ticker, save_news
    from ingestion.embedder import ingest_documents

    kap_docs, news_docs, broker_docs = [], [], []

    if req.include_kap:
        kap_docs = fetch_disclosures_for_ticker(req.ticker, days_back=req.days_back)
        if kap_docs:
            save_disclosures(kap_docs, req.ticker)
            ingest_documents(kap_docs)

    if req.include_news:
        news_docs = fetch_news_for_ticker(req.ticker, days_back=req.days_back)
        if news_docs:
            save_news(news_docs, req.ticker)
            ingest_documents(news_docs)

    total = len(kap_docs) + len(news_docs) + len(broker_docs)

    return IngestResponse(
        ticker=req.ticker,
        documents_ingested=total,
        kap_count=len(kap_docs),
        news_count=len(news_docs),
        brokerage_count=len(broker_docs),
        message=(
            f"Successfully ingested {total} documents for {req.ticker}. "
            "To ingest brokerage PDFs, place them in data/raw/pdfs/ and call /ingest/pdfs."
        ),
    )


@app.post("/ingest/pdfs", summary="Parse and ingest brokerage PDFs from data/raw/pdfs/")
def ingest_pdfs():
    """Parse all PDFs in data/raw/pdfs/ and embed into ChromaDB."""
    from ingestion.pdf_parser import parse_all_pdfs
    from ingestion.embedder import ingest_documents

    docs = parse_all_pdfs()
    count = ingest_documents(docs)
    return {"message": f"Ingested {count} PDF chunks into brokerage_reports collection."}


@app.post("/evaluate", summary="Run the evaluation pipeline")
def evaluate(max_questions: int = 5, use_llm_metrics: bool = True):
    """Run the BIST-specific evaluation on the agent. Slow – use sparingly."""
    from agent.graph import run_agent
    from evaluation.evaluator import run_full_evaluation, summarise_results, save_results

    results = run_full_evaluation(
        agent_run_fn=lambda question, ticker: run_agent(question, ticker),
        use_llm_metrics=use_llm_metrics,
        max_questions=max_questions,
    )
    summary = summarise_results(results)
    save_results(results)
    return {"summary": summary, "n_evaluated": len(results)}


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
