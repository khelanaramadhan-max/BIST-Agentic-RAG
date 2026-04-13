# 📊 BIST Equity Intelligence Agent

## 🚀 Live Demo & Project Links
- **Live Site**: [BIST Agentic RAG on Render](https://bist-agentic-rag-1.onrender.com) *(Update with your live URL if different)*
- **Front End**: Built with Vanilla HTML/CSS/JS. Features a highly responsive UI, glowing micro-animations, glassmorphism, and live simulated market ticker integrations.
- **Back End**: Python-based FastAPI server, handling RAG orchestration and LangGraph agent logic. Deployed via **Render**.
- **Vector Database**: ChromaDB for localized embedded document retrieval.
- **LLM Engine**: Groq Llama 3.3 for ultra-low latency inference and intelligence.

---

## 1. Project Overview & Assignment Objectives
This project was developed to construct a robust, **Agentic Retrieval-Augmented Generation (RAG)** system specifically engineered for the Turkish equity market (BIST). 
The core assignment was to create an intelligent assistant that doesn't just retrieve documents, but acts as a dynamic market researcher capable of synthesizing data from multiple data feeds without issuing financial advice.

### 1.1 Core System Capabilities
- **KAP Understanding**: Automated ingestion and semantic interpretation of official Public Disclosure Platform (KAP) filings.
- **Brokerage Intelligence**: Processing of unstructured PDF research reports from leading financial institutions.
- **News Intelligence**: Real-time analysis of financial news portals and sector-specific announcements.
- **Web Search Integration**: Fallback web search capabilities via DuckDuckGo for generalized and current event queries.
- **Answer Generation**: Every response is Evidence-based, Source-cited, and Time-aware.

### 1.2 Ethical Alignment (Hard Rule)
The system is strictly governed by a non-advisory alignment layer. By design, it **cannot**:
- Provide investment advice or buy/sell signals.
- Predict future prices or percentage returns.
- Enforce valuation targets.

---

## 2. Technology Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Front End** | Vanilla HTML, CSS, JavaScript (Dynamic UI, WebSocket/REST API integration) |
| **Back End** | FastAPI (Python) hosted on Render |
| **LLM Engine** | Llama 3.3 70B via Groq |
| **Frameworks** | LangChain, LangGraph (Agent State Machine) |
| **Vector DB** | ChromaDB (Persistent Storage) |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (Turkish Optimized) |
| **Extraction** | BeautifulSoup4 (KAP/News), PyMuPDF (PDFs) |

---

## 3. Agentic RAG Architecture & Process Pipeline
The intelligence agent operates via a self-correcting state machine implemented in **LangGraph**. The full process runs as follows:

1. **User Query Intake**: The user submits a question through the responsive UI.
2. **Source Router**: The LLM evaluates the query to dynamically formulate a strategy—deciding whether to query KAP, News, Brokerage reports, or Live Web search.
3. **Retrieval & Context Generation**: Embeddings are queried in the ChromaDB vector tables or live searches are performed.
4. **Context Grader**: The retrieved data is graded. If the initial retrieval is insufficient, the **Rewriter** node optimizes the query for a second pass.
5. **Cross-Source Verification (Consistency Logic)**: The agent detects discrepancies between official disclosures (KAP) and media narratives, balancing different sources.
6. **Final Synthesis**: Generates a consolidated, well-structured report summarizing all aspects directly requested by the user.
7. **Guardrails Application**: The generated text is passed through the compliance layer ensuring no investment advice is given, and appending the mandatory disclaimer.

---

## 4. Evaluation Report
The system was validated against a dataset of **12 BIST-specific questions** covering four mandatory assignment categories:

1. **KAP-Centric**: "What types of KAP disclosures has ASELS published in the last 6 months?"
2. **Brokerage Narrative**: "What common themes appear across recent reports for AKBNK?"
3. **Consistency Analysis**: "Do recent news articles contradict or align with official KAP disclosures for BIMAS?"
4. **Narrative Evolution**: "How has the tone of news about PGSUS changed compared to last year?"

**Key Metrics Performance:**
- **Faithfulness**: ~0.90
- **Answer Relevancy**: ~0.88
- **Source Coverage**: ~0.85
- **Non-Advisory Rate**: 1.00 (Mandatory)

---

## 5. Getting Started Locally

### Quick Start with Docker
```bash
docker compose up -d --build
```
*Note: Make sure to rename `.env.example` to `.env` and fill in your API keys (like Groq) before running.*

### Running the API Directly
```bash
pip install -r requirements-api.txt
python -m api.main
```
Then visit: `http://localhost:8080/` to view the UI.

---
*Disclaimer: This system does not provide investment advice.*
