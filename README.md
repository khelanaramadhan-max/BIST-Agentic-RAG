<div align="center">

# 📊 BIST Agentic Intelligence

<br>

![Agentic RAG](https://img.shields.io/badge/Agentic_RAG-Powered_by_Groq-blue?style=for-the-badge&logo=rocket)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) 
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Storage-orange?style=for-the-badge)

*An advanced AI system engineered to uncover hidden alpha, decode complex market narratives, and synthesize massive financial data within the Turkish Equity Markets (Borsa İstanbul).*

[Explore the Interactive UI Locally](http://localhost:8501) | [View API Documentation](http://localhost:8000/docs)

---

</div>

## ✨ The Vision

Traditional stock screening relies on static numbers. **BIST Agentic Intelligence** operates on *narratives*. 

By deploying a synchronized LangGraph agent workflow backed by a lightning-fast Groq LLaMA-3.3 70B model, this RAG system autonomously reads, analyzes, and cross-references thousands of local financial data points. It is designed to empower analysts and traders with institutional-grade insights instantly.

## 🧠 How it Works

When you ask a question, the Intelligence Agent:
1. **Routes intelligently**: Decides whether your question requires official regulatory filings (KAP), live financial news, or deep brokerage research PDFs.
2. **Retrieves precisely**: Scans embedded vectors in ChromaDB to extract exact paragraphs matching mathematical semantics.
3. **Grades autonomously**: A secondary AI "Grader" evaluates if the retrieved text actually answers the question. If not, the query is rewritten and searched again.
4. **Synthesizes flawlessly**: Output is generated with mandatory date-citations, source-awareness, and strict alignment to BIST mechanics.
5. **Logs analytics**: Every interaction is logged to Supabase to build institutional memory.

---

## ⚡ Supercharged Features

- **Multi-Source Triage**: Instantly searches across KAP (Kamu Aydınlatma Platformu) disclosures, Turkish financial news RSS feeds, and Brokerage PDFs simultaneously.
- **Agentic Loop**: Self-correcting AI that actively rephrases and re-searches if the initial data was insufficient.
- **Local Embedded Storage**: Heavyweight Vector DB locally hosted via ChromaDB.
- **Ultra-Low Latency Inference**: Powered by Groq’s LPU technology meaning 70-billion parameter reasoning happens in milliseconds.
- **Cross-Source Consistency**: It actively detects if the News narrative contradicts the official KAP announcement.

---

## 🚀 Getting Started

### Run with Docker Compose (Recommended)
You only need one command to spin up the entire intelligence ecosystem!

```bash
docker-compose up -d --build
```

**Services Launched:**
- **Frontend App**: Interactive Streamlit GUI natively hosted at `http://localhost:8501`.
- **Backend API**: The FastAPI intelligence core at `http://localhost:8000`.
- **Docs/Swagger**: Explore the `/query` endpoints at `http://localhost:8000/docs`.

### Deploying the HTML Frontend
This repository deploys the chat UI from the `ui` folder via **GitHub Actions** (workflow *Deploy UI to GitHub Pages*).

1. In the repo go to **Settings → Pages → Build and deployment** and set **Source** to **GitHub Actions** (not “Deploy from a branch”). Otherwise GitHub may show the **README** instead of the app.
2. Push to `main`; the workflow publishes the UI at your Pages URL (the interactive chat loads at the site root).

If you must use **Deploy from branch** with the repository root, open **`/ui/index.html`**, or use the root **`index.html`** in this repo (it redirects to `ui/index.html`).

---

## 🔌 Using the API (Swagger)

The system exposes a rich REST interface. You can natively query the agent:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
           "question": "What is the common theme in the latest ASELS research reports?",
           "ticker": "ASELS",
           "chat_history": []
         }'
```

Every response yields the answer alongside dense metadata (iteration counts, confidence scores, and raw sources).

---

## 📁 Repository Map

- `/api` - The FastAPI endpoint routes matching the RAG system.
- `/agent` - LangChain schemas, routing logic, and system prompts.
- `/ingestion` - The massive data scrapers for news, KAP, and PDF parsing.
- `/vectordb` - ChromaDB persistent storage interface.
- `/ui` - Both the Streamlit dashboards and standalone HTML clients.

<div align="center">
<i>Built to find the signal in the noise.</i>
</div>
