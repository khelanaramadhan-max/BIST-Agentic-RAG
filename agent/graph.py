"""
LangGraph Agentic RAG Graph.
Implements the retrieve → grade → re-retrieve → answer loop.

State machine nodes:
  route_query → retrieve → grade_context → [answer | rewrite → retrieve] → guardrail → END
  route_query → general_chat → guardrail → END
  (fallback paths when retrieval fails)
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, TypedDict, Literal

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
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
    GENERAL_CHAT_PROMPT,
    FALLBACK_ANSWER_PROMPT,
    DISCLAIMER,
)
from agent.tools import (
    search_kap_disclosures,
    search_financial_news,
    search_brokerage_reports,
    search_all_sources,
    search_live_web,
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
    original_question: str  # Preserved original for fallback
    ticker: str

    # Routing
    selected_sources: list[str]  # ["kap", "news", "brokerage", "web"]
    routing_reasoning: str
    interaction_mode: str  # "market" | "general"

    # Retrieval
    kap_context: str
    news_context: str
    brokerage_context: str
    web_context: str
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
    answer_type: str  # "direct" | "consistency" | "narrative" | "general" | "fallback"


# ─── LLM Setup ───────────────────────────────────────────────────────────────


def _get_llm(temperature: float = 0.0) -> ChatGroq | ChatOpenAI:
    if settings.use_openai():
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=temperature,
            max_tokens=4096,
        )
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
        mode_raw = str(data.get("mode", "market")).strip().lower()
        interaction_mode = "general" if mode_raw == "general" else "market"

        if interaction_mode == "general":
            sources = []
        else:
            sources = data.get("sources", ["web"])
            if not sources:
                # If mode is market but no sources specified, at least use web
                sources = ["web"]

        reasoning = data.get("reasoning", "")
        ticker = data.get("ticker", state.get("ticker", ""))

        logger.info(
            "Router: mode=%s, sources=%s, ticker=%s",
            interaction_mode,
            sources,
            ticker,
        )

        # Determine answer type
        question_lower = state["question"].lower()
        if any(k in question_lower for k in ["consistent", "contradict", "align", "compare"]):
            answer_type = "consistency"
        elif any(k in question_lower for k in ["evolution", "changed", "trend", "narrative", "over time"]):
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
            "interaction_mode": interaction_mode,
            "original_question": state["question"],
        }
    except Exception as exc:
        logger.warning("Router error: %s – defaulting to web + all sources", exc)
        return {
            **state,
            "selected_sources": ["kap", "news", "brokerage", "web"],
            "routing_reasoning": "Fallback: query all sources including web",
            "iteration_count": state.get("iteration_count", 0),
            "answer_type": "direct",
            "interaction_mode": "market",
            "original_question": state["question"],
        }


def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieve from selected sources. Each source is independently try/excepted."""
    question = state["question"]
    ticker = state.get("ticker", "")
    sources = state.get("selected_sources", ["web"])

    kap_ctx = ""
    news_ctx = ""
    broker_ctx = ""
    web_ctx = ""

    if "kap" in sources:
        try:
            kap_ctx = search_kap_disclosures.invoke(
                {"query": question, "ticker": ticker}
            )
        except Exception as exc:
            logger.warning("KAP retrieval failed: %s", exc)
            kap_ctx = "No documents found."

    if "news" in sources:
        try:
            news_ctx = search_financial_news.invoke(
                {"query": question, "ticker": ticker}
            )
        except Exception as exc:
            logger.warning("News retrieval failed: %s", exc)
            news_ctx = "No documents found."

    if "brokerage" in sources:
        try:
            broker_ctx = search_brokerage_reports.invoke(
                {"query": question, "ticker": ticker}
            )
        except Exception as exc:
            logger.warning("Brokerage retrieval failed: %s", exc)
            broker_ctx = "No documents found."

    if "web" in sources:
        try:
            web_ctx = search_live_web.invoke(
                {"query": question, "ticker": ticker}
            )
        except Exception as exc:
            logger.warning("Web retrieval failed: %s", exc)
            web_ctx = "Web search unavailable."

    # Build combined context — only include sections that have real content
    sections = []
    if kap_ctx and "No documents found" not in kap_ctx:
        sections.append(f"=== KAP DISCLOSURES ===\n{kap_ctx}")
    if news_ctx and "No documents found" not in news_ctx:
        sections.append(f"=== FINANCIAL NEWS ===\n{news_ctx}")
    if broker_ctx and "No documents found" not in broker_ctx:
        sections.append(f"=== BROKERAGE REPORTS ===\n{broker_ctx}")
    if web_ctx and "No relevant web results" not in web_ctx and "unavailable" not in web_ctx:
        sections.append(f"=== WEB SEARCH RESULTS ===\n{web_ctx}")

    combined = "\n\n".join(sections) if sections else ""

    # Track which sources actually returned useful data
    sources_used = []
    if kap_ctx and "No documents found" not in kap_ctx:
        sources_used.append("kap")
    if news_ctx and "No documents found" not in news_ctx:
        sources_used.append("news")
    if broker_ctx and "No documents found" not in broker_ctx:
        sources_used.append("brokerage")
    if web_ctx and "No relevant web results" not in web_ctx and "unavailable" not in web_ctx:
        sources_used.append("web")

    logger.info(
        "Retrieved context: kap=%d chars, news=%d chars, brokerage=%d chars, web=%d chars | useful_sources=%s",
        len(kap_ctx), len(news_ctx), len(broker_ctx), len(web_ctx), sources_used,
    )

    return {
        **state,
        "kap_context": kap_ctx,
        "news_context": news_ctx,
        "brokerage_context": broker_ctx,
        "web_context": web_ctx,
        "combined_context": combined,
        "sources_used": sources_used,
    }


def grade_context(state: AgentState) -> AgentState:
    """Evaluate whether the retrieved context is sufficient to answer the question."""
    context = state.get("combined_context", "")
    
    # If there's no context at all, skip grading and go straight to fallback
    if not context.strip():
        logger.info("Grader: no context retrieved — will use fallback answer")
        return {
            **state,
            "context_sufficient": True,  # Let it through to answer node which handles empty context
            "grader_confidence": 0.1,
            "missing_aspects": ["No context retrieved from any source"],
            "rewrite_hint": "",
        }

    # If we have web context, be very lenient — web results are almost always useful enough
    if state.get("web_context", "") and "No relevant" not in state.get("web_context", ""):
        logger.info("Grader: web context available — marking as sufficient")
        return {
            **state,
            "context_sufficient": True,
            "grader_confidence": 0.8,
            "missing_aspects": [],
            "rewrite_hint": "",
        }

    llm = _get_llm()
    chain = GRADER_PROMPT | llm

    try:
        response = chain.invoke(
            {"question": state["question"], "context": context[:4000]}
        )
        raw = response.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        data = json.loads(raw)
        sufficient = data.get("sufficient", True)
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
        logger.warning("Grader error: %s — passing through", exc)
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
    """Generate the final answer using the appropriate prompt based on context availability."""
    llm = _get_llm(temperature=0.2)
    answer_type = state.get("answer_type", "direct")
    context = state.get("combined_context", "")
    question = state.get("original_question", state["question"])

    try:
        if answer_type == "consistency":
            chain = CONSISTENCY_PROMPT | llm
            response = chain.invoke(
                {
                    "ticker": state.get("ticker", ""),
                    "question": question,
                    "kap_context": state.get("kap_context", "N/A"),
                    "news_context": state.get("news_context", "N/A"),
                    "brokerage_context": state.get("brokerage_context", "N/A"),
                }
            )
        elif not context.strip():
            # No context at all — use fallback prompt (pure LLM knowledge)
            logger.info("Using fallback answer (no context available)")
            chain = FALLBACK_ANSWER_PROMPT | llm
            response = chain.invoke(
                {
                    "question": question,
                    "chat_history": state.get("chat_history", []),
                }
            )
            answer_type = "fallback"
        else:
            # Normal RAG or web-augmented answer
            chain = ANSWER_PROMPT | llm
            response = chain.invoke(
                {
                    "question": question,
                    "context": context,
                    "chat_history": state.get("chat_history", []),
                }
            )

        answer = response.content.strip()
        logger.info("Answer generated (%d chars, type=%s)", len(answer), answer_type)

        return {
            **state,
            "final_answer": answer,
            "answer_type": answer_type,
            "chat_history": [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ],
        }
    except Exception as exc:
        logger.error("Answer generation error: %s", exc)
        
        # Last resort: try a simpler prompt
        try:
            simple_llm = _get_llm(temperature=0.3)
            from langchain_core.prompts import ChatPromptTemplate
            emergency_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Answer the user's question to the best of your ability. Always respond in English."),
                ("human", "{question}"),
            ])
            chain = emergency_prompt | simple_llm
            response = chain.invoke({"question": question})
            answer = response.content.strip()
            return {
                **state,
                "final_answer": answer,
                "answer_type": "fallback",
            }
        except Exception as exc2:
            logger.error("Emergency answer also failed: %s", exc2)
            err_answer = (
                f"I encountered an error processing your question. "
                f"Error: {exc}\n\n"
                f"Please try rephrasing your question or check that the API keys are correctly configured."
            )
            return {**state, "final_answer": err_answer, "answer_type": "error"}


def apply_guardrail_node(state: AgentState) -> AgentState:
    """Apply ethical/compliance guardrails to the generated answer."""
    safe_answer = apply_guardrails(state.get("final_answer", ""))
    return {**state, "final_answer": safe_answer}


def general_chat(state: AgentState) -> AgentState:
    """Answer without RAG (greetings, small talk, general pleasantries)."""
    llm = _get_llm(temperature=0.55)
    chain = GENERAL_CHAT_PROMPT | llm
    try:
        response = chain.invoke({"question": state["question"]})
        answer = response.content.strip()
    except Exception as exc:
        logger.error("General chat error: %s", exc)
        answer = (
            f"I'm having trouble connecting to the language model ({exc}). "
            "Please check that GROQ_API_KEY or OPENAI_API_KEY is set in the .env file."
        )
    return {
        **state,
        "final_answer": answer,
        "sources_used": [],
        "answer_type": "general",
        "grader_confidence": 1.0,
        "iteration_count": 0,
    }


# ─── Routing Logic ────────────────────────────────────────────────────────────


def route_after_router(state: AgentState) -> Literal["retrieve", "general_chat"]:
    if state.get("interaction_mode") == "general":
        return "general_chat"
    return "retrieve"


def should_rewrite(state: AgentState) -> Literal["rewrite", "answer"]:
    """Decide whether to re-retrieve or generate final answer."""
    # If context is sufficient, go to answer
    if state.get("context_sufficient", True):
        return "answer"
    
    # If we've exhausted retries, go to answer anyway (we'll use fallback)
    if state.get("iteration_count", 0) >= settings.max_retrieval_iterations:
        logger.info("Max iterations reached — generating answer with available context")
        return "answer"
    
    logger.info("Context insufficient – rewriting query (iteration %d)", state.get("iteration_count", 0))
    return "rewrite"


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
    graph.add_node("general_chat", general_chat)

    # Define edges
    graph.set_entry_point("route_query")
    graph.add_conditional_edges(
        "route_query",
        route_after_router,
        {"retrieve": "retrieve", "general_chat": "general_chat"},
    )
    graph.add_edge("retrieve", "grade")

    # Conditional: sufficient → answer, else → rewrite
    graph.add_conditional_edges(
        "grade",
        should_rewrite,
        {"rewrite": "rewrite", "answer": "answer"},
    )

    graph.add_edge("rewrite", "retrieve")  # loop back
    graph.add_edge("answer", "guardrail")
    graph.add_edge("general_chat", "guardrail")
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
        "original_question": question,
        "ticker": ticker,
        "selected_sources": [],
        "routing_reasoning": "",
        "kap_context": "",
        "news_context": "",
        "brokerage_context": "",
        "web_context": "",
        "combined_context": "",
        "context_sufficient": False,
        "grader_confidence": 0.0,
        "missing_aspects": [],
        "rewrite_hint": "",
        "iteration_count": 0,
        "final_answer": "",
        "sources_used": [],
        "answer_type": "direct",
        "interaction_mode": "market",
    }

    logger.info("Agent running: question='%s', ticker='%s'", question[:80], ticker)
    
    try:
        final_state = agent.invoke(initial_state)
    except Exception as exc:
        logger.error("Graph execution error: %s", exc, exc_info=True)
        # Emergency fallback: try direct LLM call
        try:
            llm = _get_llm(temperature=0.3)
            from langchain_core.prompts import ChatPromptTemplate
            emergency = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Answer the user's question thoroughly and in English."),
                ("human", "{question}"),
            ])
            resp = (emergency | llm).invoke({"question": question})
            return {
                "answer": resp.content.strip(),
                "sources_used": [],
                "iteration_count": 0,
                "answer_type": "fallback",
                "routing_reasoning": f"Emergency fallback due to: {exc}",
                "grader_confidence": 0.0,
            }
        except Exception as exc2:
            return {
                "answer": f"I'm sorry, I encountered an error: {exc}. Please check your API configuration.",
                "sources_used": [],
                "iteration_count": 0,
                "answer_type": "error",
                "routing_reasoning": f"Error: {exc}",
                "grader_confidence": 0.0,
            }

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
    q = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
    ticker = sys.argv[2] if len(sys.argv) > 2 else ""
    result = run_agent(q, ticker)
    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print(f"\nSources: {result['sources_used']}")
    print(f"Iterations: {result['iteration_count']}")
    print(f"Type: {result['answer_type']}")
