"""
LangGraph Agentic RAG Graph.
Implements the retrieve → grade → re-retrieve → answer loop.

State machine nodes:
  route_query → retrieve → grade_context → [answer | rewrite → retrieve] → guardrail → END
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, TypedDict, Literal

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from config.settings import settings
from agent.prompts import (
    ROUTER_PROMPT,
    GRADER_PROMPT,
    REWRITER_PROMPT,
    ANSWER_PROMPT,
    CONSISTENCY_PROMPT,
    DISCLAIMER,
)
from agent.tools import (
    search_kap_disclosures,
    search_financial_news,
    search_brokerage_reports,
    search_all_sources,
    ALL_TOOLS,
)
from guardrails.checker import apply_guardrails

logger = logging.getLogger(__name__)


# ─── State Definition ─────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """Complete state that flows through the LangGraph nodes."""

    # Conversation
    chat_history: Annotated[list[BaseMessage], add_messages]
    question: str
    ticker: str

    # Routing
    selected_sources: list[str]  # ["kap", "news", "brokerage"]
    routing_reasoning: str

    # Retrieval
    kap_context: str
    news_context: str
    brokerage_context: str
    combined_context: str

    # Grading
    context_sufficient: bool
    grader_confidence: float
    missing_aspects: list[str]
    rewrite_hint: str

    # Iteration control
    iteration_count: int

    # Output
    final_answer: str
    sources_used: list[str]
    answer_type: str  # "direct" | "consistency" | "narrative"


# ─── LLM Setup ───────────────────────────────────────────────────────────────


def _get_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=temperature,
        max_tokens=4096,
    )


# ─── Nodes ───────────────────────────────────────────────────────────────────


def route_query(state: AgentState) -> AgentState:
    """Decide which sources to query based on the question type."""
    llm = _get_llm()
    chain = ROUTER_PROMPT | llm

    try:
        response = chain.invoke({"question": state["question"]})
        raw = response.content.strip()

        # Extract JSON from response
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        data = json.loads(raw)
        sources = data.get("sources", ["kap", "news"])
        reasoning = data.get("reasoning", "")
        ticker = data.get("ticker", state.get("ticker", ""))

        logger.info("Router: sources=%s, ticker=%s", sources, ticker)

        # Determine answer type
        question_lower = state["question"].lower()
        if any(k in question_lower for k in ["tutarlı", "çelişki", "consistent", "contradict", "align"]):
            answer_type = "consistency"
        elif any(k in question_lower for k in ["değeşim", "evolution", "changed", "trend", "narrative"]):
            answer_type = "narrative"
        else:
            answer_type = "direct"

        return {
            **state,
            "selected_sources": sources,
            "routing_reasoning": reasoning,
            "ticker": ticker,
            "answer_type": answer_type,
            "iteration_count": state.get("iteration_count", 0),
        }
    except Exception as exc:
        logger.warning("Router error: %s – defaulting to all sources", exc)
        return {
            **state,
            "selected_sources": ["kap", "news", "brokerage"],
            "routing_reasoning": "Fallback: query all sources",
            "iteration_count": state.get("iteration_count", 0),
            "answer_type": "direct",
        }


def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieve from selected sources in parallel."""
    question = state["question"]
    ticker = state.get("ticker", "")
    sources = state.get("selected_sources", ["kap", "news", "brokerage"])

    kap_ctx = ""
    news_ctx = ""
    broker_ctx = ""

    if "kap" in sources:
        try:
            kap_ctx = search_kap_disclosures.invoke(
                {"query": question, "ticker": ticker}
            )
        except Exception as exc:
            logger.warning("KAP retrieval failed: %s", exc)
            kap_ctx = "KAP verisi alınamadı."

    if "news" in sources:
        try:
            news_ctx = search_financial_news.invoke(
                {"query": question, "ticker": ticker}
            )
        except Exception as exc:
            logger.warning("News retrieval failed: %s", exc)
            news_ctx = "Haber verisi alınamadı."

    if "brokerage" in sources:
        try:
            broker_ctx = search_brokerage_reports.invoke(
                {"query": question, "ticker": ticker}
            )
        except Exception as exc:
            logger.warning("Brokerage retrieval failed: %s", exc)
            broker_ctx = "Araştırma raporu verisi alınamadı."

    combined = "\n\n".join(
        filter(None, [
            f"=== KAP AÇIKLAMALARI ===\n{kap_ctx}" if kap_ctx else "",
            f"=== FİNANSAL HABERLER ===\n{news_ctx}" if news_ctx else "",
            f"=== ARACI KURUM RAPORLARI ===\n{broker_ctx}" if broker_ctx else "",
        ])
    )

    sources_used = [s for s in sources if (
        (s == "kap" and "No documents" not in kap_ctx)
        or (s == "news" and "No documents" not in news_ctx)
        or (s == "brokerage" and "No documents" not in broker_ctx)
    )]

    logger.info(
        "Retrieved context: kap=%d chars, news=%d chars, brokerage=%d chars",
        len(kap_ctx), len(news_ctx), len(broker_ctx),
    )

    return {
        **state,
        "kap_context": kap_ctx,
        "news_context": news_ctx,
        "brokerage_context": broker_ctx,
        "combined_context": combined,
        "sources_used": sources_used,
    }


def grade_context(state: AgentState) -> AgentState:
    """Evaluate whether the retrieved context is sufficient to answer the question."""
    llm = _get_llm()
    chain = GRADER_PROMPT | llm

    context = state.get("combined_context", "")
    if not context.strip() or context == "No documents found.":
        return {
            **state,
            "context_sufficient": False,
            "grader_confidence": 0.0,
            "missing_aspects": ["No context retrieved"],
            "rewrite_hint": f"{state['question']} {state.get('ticker', '')} BIST Turkey",
        }

    try:
        response = chain.invoke(
            {"question": state["question"], "context": context[:3000]}
        )
        raw = response.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        data = json.loads(raw)
        sufficient = data.get("sufficient", False)
        confidence = float(data.get("confidence", 0.5))
        missing = data.get("missing_aspects", [])
        hint = data.get("rewrite_hint", "")

        logger.info(
            "Grader: sufficient=%s, confidence=%.2f, missing=%s",
            sufficient, confidence, missing,
        )

        return {
            **state,
            "context_sufficient": sufficient,
            "grader_confidence": confidence,
            "missing_aspects": missing,
            "rewrite_hint": hint,
        }
    except Exception as exc:
        logger.warning("Grader error: %s", exc)
        return {
            **state,
            "context_sufficient": True,  # Pass through on error
            "grader_confidence": 0.5,
            "missing_aspects": [],
            "rewrite_hint": "",
        }


def rewrite_query(state: AgentState) -> AgentState:
    """Rewrite the query to improve retrieval in the next iteration."""
    llm = _get_llm()
    chain = REWRITER_PROMPT | llm

    try:
        response = chain.invoke(
            {
                "question": state["question"],
                "missing_aspects": ", ".join(state.get("missing_aspects", [])),
            }
        )
        new_question = response.content.strip()
        logger.info("Query rewritten: '%s' → '%s'", state["question"][:60], new_question[:60])
        return {
            **state,
            "question": new_question,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    except Exception as exc:
        logger.warning("Rewriter error: %s", exc)
        return {
            **state,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }


def generate_answer(state: AgentState) -> AgentState:
    """Generate the final answer using ANSWER_PROMPT or CONSISTENCY_PROMPT."""
    llm = _get_llm(temperature=0.1)
    answer_type = state.get("answer_type", "direct")

    try:
        if answer_type == "consistency":
            chain = CONSISTENCY_PROMPT | llm
            response = chain.invoke(
                {
                    "ticker": state.get("ticker", ""),
                    "question": state["question"],
                    "kap_context": state.get("kap_context", "N/A"),
                    "news_context": state.get("news_context", "N/A"),
                    "brokerage_context": state.get("brokerage_context", "N/A"),
                }
            )
        else:
            chain = ANSWER_PROMPT | llm
            response = chain.invoke(
                {
                    "question": state["question"],
                    "context": state.get("combined_context", ""),
                    "chat_history": state.get("chat_history", []),
                }
            )

        answer = response.content.strip()
        logger.info("Answer generated (%d chars)", len(answer))

        return {
            **state,
            "final_answer": answer,
            "chat_history": [
                HumanMessage(content=state["question"]),
                AIMessage(content=answer),
            ],
        }
    except Exception as exc:
        logger.error("Answer generation error: %s", exc)
        err_answer = (
            f"Bu soruyu yanıtlarken bir hata oluştu: {exc}\n\n"
            f"Lütfen sorunuzu tekrar formüle edip deneyin.\n\n{DISCLAIMER}"
        )
        return {**state, "final_answer": err_answer}


def apply_guardrail_node(state: AgentState) -> AgentState:
    """Apply ethical/compliance guardrails to the generated answer."""
    safe_answer = apply_guardrails(state.get("final_answer", ""))
    return {**state, "final_answer": safe_answer}


# ─── Routing Logic ────────────────────────────────────────────────────────────


def should_rewrite(state: AgentState) -> Literal["rewrite", "answer"]:
    """Decide whether to re-retrieve or generate final answer."""
    if not state.get("context_sufficient", True):
        if state.get("iteration_count", 0) < settings.max_retrieval_iterations:
            logger.info("Context insufficient – rewriting query (iteration %d)", state["iteration_count"])
            return "rewrite"
    return "answer"


# ─── Graph Assembly ───────────────────────────────────────────────────────────


def build_agent_graph() -> StateGraph:
    """Build and compile the LangGraph agentic RAG graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("route_query", route_query)
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("grade", grade_context)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("answer", generate_answer)
    graph.add_node("guardrail", apply_guardrail_node)

    # Define edges
    graph.set_entry_point("route_query")
    graph.add_edge("route_query", "retrieve")
    graph.add_edge("retrieve", "grade")

    # Conditional: sufficient → answer, else → rewrite
    graph.add_conditional_edges(
        "grade",
        should_rewrite,
        {"rewrite": "rewrite", "answer": "answer"},
    )

    graph.add_edge("rewrite", "retrieve")  # loop back
    graph.add_edge("answer", "guardrail")
    graph.add_edge("guardrail", END)

    return graph.compile()


# Singleton compiled graph
_graph = None


def get_agent():
    """Return (and lazily create) the compiled agent graph."""
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph


# ─── Main Entry Point ─────────────────────────────────────────────────────────


def run_agent(
    question: str,
    ticker: str = "",
    chat_history: list[BaseMessage] | None = None,
) -> dict:
    """
    Run the agentic RAG loop for a given question.

    Returns:
        dict with keys: answer, sources_used, iteration_count, answer_type
    """
    agent = get_agent()

    initial_state: AgentState = {
        "chat_history": chat_history or [],
        "question": question,
        "ticker": ticker,
        "selected_sources": [],
        "routing_reasoning": "",
        "kap_context": "",
        "news_context": "",
        "brokerage_context": "",
        "combined_context": "",
        "context_sufficient": False,
        "grader_confidence": 0.0,
        "missing_aspects": [],
        "rewrite_hint": "",
        "iteration_count": 0,
        "final_answer": "",
        "sources_used": [],
        "answer_type": "direct",
    }

    logger.info("Agent running: question='%s', ticker='%s'", question[:80], ticker)
    final_state = agent.invoke(initial_state)

    return {
        "answer": final_state.get("final_answer", ""),
        "sources_used": final_state.get("sources_used", []),
        "iteration_count": final_state.get("iteration_count", 0),
        "answer_type": final_state.get("answer_type", "direct"),
        "routing_reasoning": final_state.get("routing_reasoning", ""),
        "grader_confidence": final_state.get("grader_confidence", 0.0),
    }


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    q = sys.argv[1] if len(sys.argv) > 1 else "ASELS son 6 ayda hangi KAP açıklamalarını yaptı?"
    ticker = sys.argv[2] if len(sys.argv) > 2 else "ASELS"
    result = run_agent(q, ticker)
    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print(f"\nSources: {result['sources_used']}")
    print(f"Iterations: {result['iteration_count']}")
