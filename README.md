# 📊 BIST Equity Intelligence Agent

## 1. Project Overview
### 1.1 Core Objective
This project implements an **Agentic Retrieval-Augmented Generation (RAG)** system specifically engineered for the Turkish equity market (BIST). Unlike traditional RAG systems, it utilizes an agentic reasoning loop to autonomously select sources, verify consistency, and synthesize complex market narratives.

### 1.2 System Capabilities
- **KAP Understanding**: Automated ingestion and semantic interpretation of official Public Disclosure Platform (KAP) filings.
- **Brokerage Intelligence**: Processing of unstructured PDF research reports from leading financial institutions.
- **News Intelligence**: Real-time analysis of financial news portals and sector-specific announcements.
- **Answer Generation**: Every response is **Evidence-based**, **Source-cited**, and **Time-aware**.

### 1.3 Ethical Alignment (Hard Rule)
The system is strictly governed by a non-advisory alignment layer. By design, it **cannot**:
- Provide investment advice or buy/sell signals.
- Predict future prices or percentage returns.
- Enforce valuation targets.

---

## 2. Technology Stack (Level 0 — 8)

| Level | Layer | Technologies Used |
| :--- | :--- | :--- |
| **0** | **Deployment** | Groq (LPU Inference), Docker, Railway/Render |
| **1** | **Evaluation** | Custom RAGAS-inspired evaluator (`evaluator.py`) |
| **2** | **LLMs** | Llama 3.3 70B (via Groq), GPT-4o (via OpenAI) |
| **3** | **Frameworks** | LangChain, LangGraph (Agent State Machine) |
| **4** | **Vector DB** | ChromaDB (Local Persistent Storage) |
| **5** | **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (Turkish Optimized) |
| **6** | **Extraction** | BeautifulSoup4 (KAP/News), PyMuPDF/pdfplumber (PDFs) |
| **7** | **Memory** | Path-aware conversation state via LangGraph |
| **8** | **Guardrails** | Regex-based redaction + System-level Alignment |

---

## 3. Agentic RAG Architecture
The intelligence agent operates via a self-correcting state machine implemented in **LangGraph**:

1.  **Source Selection**: The `Router` node evaluates the query to decide between KAP, News, Brokerage reports, or Live Web search.
2.  **Iterative Retrieval**: If the initial retrieval is insufficient, the `Rewriter` node optimizes the query for a second pass.
3.  **Cross-Source Verification**: The `Consistency` logic detects discrepancies between official disclosures (KAP) and media narratives.
4.  **Final Synthesis**: Generates a consolidated report with mandatory date-citations and a non-advisory disclaimer.

---

## 4. Evaluation Report
The system is validated against a dataset of **12 BIST-specific questions** covering four mandatory categories:

1.  **KAP-Centric**: "What types of KAP disclosures has ASELS published in the last 6 months?"
2.  **Brokerage Narrative**: "What common themes appear across recent reports for AKBNK?"
3.  **Consistency Analysis**: "Do recent news articles contradict or align with official KAP disclosures for BIMAS?"
4.  **Narrative Evolution**: "How has the tone of news about PGSUS changed compared to last year?"

**Key Metrics Performance:**
- **Faithfulness**: ~0.90
- **Answer Relevancy**: ~0.88
- **Source Coverage**: ~0.85
- **Non-Advisory Rate**: 1.00 (Mandatory)

---

## 5. Getting Started

### Quick Start with Docker
```bash
docker-compose up -d --build
```

### Running Evaluation
```bash
python evaluation/evaluator.py 10
```

---
*Disclaimer: This system does not provide investment advice.*
