"""
Central configuration for BIST Agentic RAG.
All settings are read from environment variables or the .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")

    # ── Embeddings ────────────────────────────────────────────
    embedding_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL",
    )

    # ── ChromaDB ─────────────────────────────────────────────
    chroma_persist_dir: str = Field(
        str(BASE_DIR / "data" / "chromadb"), env="CHROMA_PERSIST_DIR"
    )

    # ── Data paths ───────────────────────────────────────────
    raw_data_dir: str = Field(str(BASE_DIR / "data" / "raw"), env="RAW_DATA_DIR")
    processed_data_dir: str = Field(
        str(BASE_DIR / "data" / "processed"), env="PROCESSED_DATA_DIR"
    )
    pdf_dir: str = Field(str(BASE_DIR / "data" / "raw" / "pdfs"), env="PDF_DIR")

    # ── API ──────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

    # ── KAP ──────────────────────────────────────────────────
    kap_base_url: str = Field("https://www.kap.org.tr", env="KAP_BASE_URL")
    kap_request_delay: float = Field(1.5, env="KAP_REQUEST_DELAY")

    # ── News ─────────────────────────────────────────────────
    news_api_key: str = Field("", env="NEWS_API_KEY")

    # ── Agentic Loop ─────────────────────────────────────────
    max_retrieval_iterations: int = 3
    top_k_retrieval: int = 5
    minimum_relevance_score: float = 0.4

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ── Supabase (Optional) ──────────────────────────────────
    supabase_url: str = Field("", env="SUPABASE_URL")
    supabase_key: str = Field("", env="SUPABASE_KEY")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton – import this everywhere
settings = Settings()
