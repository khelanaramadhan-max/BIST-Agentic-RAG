"""
Streamlit UI for BIST Equity Intelligence Agent.
Run: streamlit run ui/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from datetime import datetime

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BIST Equity Intelligence Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.main { background: #0f0f1a; }

.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 8px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.stat-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
}

.stat-number { color: #667eea; font-size: 1.8rem; font-weight: 700; }
.stat-label { color: #94a3b8; font-size: 0.8rem; margin-top: 4px; }

.answer-box {
    background: rgba(102, 126, 234, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
}

.source-chip {
    display: inline-block;
    background: rgba(102, 126, 234, 0.2);
    border: 1px solid rgba(102, 126, 234, 0.4);
    color: #a5b4fc;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px;
}

.disclaimer-box {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    padding: 10px 14px;
    color: #fca5a5;
    font-size: 0.8rem;
    margin-top: 12px;
}

.ticker-chip {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 4px 14px;
    border-radius: 6px;
    letter-spacing: 1px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Header ──────────────────────────────────────────────────────────────────

col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("# 📊")
with col_title:
    st.markdown('<div class="hero-badge">BIST Intelligence Agent</div>', unsafe_allow_html=True)
    st.markdown("# Turkish Equity Intelligence")
    st.markdown("*KAP Disclosures · Financial News · Brokerage Research*")

st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    ticker_options = [
        "ASELS", "GARAN", "AKBNK", "THYAO", "BIMAS",
        "EREGL", "KCHOL", "TUPRS", "SISE", "PGSUS",
    ]
    selected_ticker = st.selectbox(
        "📌 Select Ticker",
        ["(All / Auto-detect)"] + ticker_options,
        help="Filter context to a specific BIST ticker",
    )
    ticker = "" if selected_ticker == "(All / Auto-detect)" else selected_ticker

    st.divider()
    st.markdown("### 📥 Data Ingestion")

    ingest_ticker = st.text_input("Ticker to ingest", value="ASELS")
    days_back = st.slider("Days back", 30, 365, 90)
    include_kap = st.checkbox("Include KAP disclosures", value=True)
    include_news = st.checkbox("Include news", value=True)

    if st.button("🔄 Ingest Data", type="primary"):
        with st.spinner(f"Fetching data for {ingest_ticker}..."):
            try:
                from ingestion.kap_scraper import fetch_disclosures_for_ticker, save_disclosures
                from ingestion.news_fetcher import fetch_news_for_ticker, save_news
                from ingestion.embedder import ingest_documents

                total = 0
                if include_kap:
                    kap = fetch_disclosures_for_ticker(ingest_ticker, days_back=days_back)
                    save_disclosures(kap, ingest_ticker)
                    ingest_documents(kap)
                    total += len(kap)
                if include_news:
                    news = fetch_news_for_ticker(ingest_ticker, days_back=days_back)
                    save_news(news, ingest_ticker)
                    ingest_documents(news)
                    total += len(news)

                st.success(f"✅ Ingested {total} documents for {ingest_ticker}")
            except Exception as e:
                st.error(f"Ingestion error: {e}")

    st.divider()
    st.markdown("### 📊 Collection Stats")
    if st.button("Show Stats"):
        try:
            from vectordb.chroma_store import collection_stats
            stats = collection_stats()
            for col, count in stats.items():
                st.metric(col.replace("_", " ").title(), count)
        except Exception as e:
            st.warning(f"Stats error: {e}")

    st.divider()
    st.markdown(
        """
<div class='disclaimer-box' style='background: rgba(34, 197, 94, 0.1); border-color: rgba(34, 197, 94, 0.3); color: #86efac;'>
💡 <b>Pro Tip</b><br>
Ask our agent complex narrative analysis!
Dive deep into BIST dynamics.
</div>
""",
        unsafe_allow_html=True,
    )

# ─── Main Chat Interface ──────────────────────────────────────────────────────

# Quick question templates
st.markdown("### 💬 Ask the Agent")

SAMPLE_QUESTIONS = [
    "What KAP disclosures did ASELS make in the last 6 months?",
    "What are the common themes in GARAN research reports?",
    "Is BIMAS news consistent with official KAP disclosures?",
    "How did the narrative around THYAO change?",
    "Latest statements on the Turkish banking sector?",
]

col1, col2, col3 = st.columns(3)
for i, q in enumerate(SAMPLE_QUESTIONS):
    col = [col1, col2, col3][i % 3]
    with col:
        if st.button(f"💭 {q[:45]}...", key=f"sample_{i}", use_container_width=True):
            st.session_state["pending_question"] = q

# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Iterations", meta.get("iteration_count", 0))
            c2.metric("Confidence", f"{meta.get('grader_confidence', 0):.0%}")
            c3.metric("Sources", ", ".join(meta.get("sources_used", [])) or "—")

# ── User input ────────────────────────────────────────────────────────────────

pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Ask about a BIST company or market event...") or pending

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching KAP, news, and brokerage reports..."):
            try:
                from agent.graph import run_agent

                result = run_agent(
                    question=user_input,
                    ticker=ticker,
                    chat_history=[],
                )

                answer = result.get("answer", "Failed to generate answer.")
                st.markdown(answer)

                # Metadata chips
                meta = {
                    "sources_used": result.get("sources_used", []),
                    "iteration_count": result.get("iteration_count", 0),
                    "grader_confidence": result.get("grader_confidence", 0.0),
                    "answer_type": result.get("answer_type", "direct"),
                }
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🔄 Iterations", meta["iteration_count"])
                c2.metric("📊 Confidence", f"{meta['grader_confidence']:.0%}")
                c3.metric("🎯 Type", meta["answer_type"].title())
                c4.metric("📚 Sources", len(meta["sources_used"]))

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "meta": meta}
                )

            except Exception as exc:
                err = f"❌ Agent error: {exc}\n\nPlease ensure ChromaDB is populated. Run: `python ingest_pipeline.py`"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

# ─── Evaluation Panel ─────────────────────────────────────────────────────────

with st.expander("🧪 Run Evaluation", expanded=False):
    st.markdown("Run the BIST-specific evaluation pipeline on the agent.")
    max_q = st.slider("Number of questions", 1, 12, 3)
    use_llm = st.checkbox("Use LLM-based metrics (slower, more accurate)", value=True)

    if st.button("▶️ Run Evaluation"):
        with st.spinner(f"Evaluating {max_q} questions..."):
            try:
                from agent.graph import run_agent
                from evaluation.evaluator import run_full_evaluation, summarise_results

                results = run_full_evaluation(
                    agent_run_fn=lambda question, ticker: run_agent(question, ticker),
                    use_llm_metrics=use_llm,
                    max_questions=max_q,
                )
                summary = summarise_results(results)

                st.success(f"✅ Evaluated {len(results)} questions")
                st.json(summary)

                import pandas as pd
                df = pd.DataFrame([
                    {
                        "ID": r.question_id,
                        "Category": r.category,
                        "Faithfulness": f"{r.faithfulness:.2f}",
                        "Relevancy": f"{r.answer_relevancy:.2f}",
                        "Source Coverage": f"{r.source_coverage:.2f}",
                        "Disclaimer": "✅" if r.disclaimer_present else "❌",
                        "Non-Advisory": "✅" if r.non_advisory else "❌",
                        "Overall": f"{r.overall_score:.2f}",
                    }
                    for r in results
                ])
                st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Evaluation error: {e}")

# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    f"""
<div style='text-align:center; color:#64748b; font-size:0.8rem; padding:10px 0'>
    BIST Equity Intelligence Agent • Powered by Groq + LangGraph + ChromaDB<br>
    ✨ Discover hidden alpha in Turkish markets.<br>
    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""",
    unsafe_allow_html=True,
)
