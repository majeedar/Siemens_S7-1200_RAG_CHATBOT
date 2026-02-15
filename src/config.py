"""Centralized configuration for the S7-1200 RAG Chatbot."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    PDF_PATH: str = ""

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "s7_1200_manual"
    QDRANT_VECTOR_SIZE: int = 384

    # Ollama
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b-instruct-q4_K_M"
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_TOP_P: float = 0.9
    OLLAMA_NUM_CTX: int = 4096
    OLLAMA_REPEAT_PENALTY: float = 1.1

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Retrieval
    RETRIEVAL_TOP_K: int = 5
    ENABLE_RERANKING: bool = True
    ENABLE_HYBRID_SEARCH: bool = True

    # Conversation
    MAX_HISTORY_LENGTH: int = 10

    # Monitoring
    METRICS_PORT: int = 9090
    ENABLE_METRICS: bool = True
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "s7-1200-rag"
    LOW_SCORE_THRESHOLD: float = 0.3

    # Gradio
    GRADIO_SERVER_NAME: str = "127.0.0.1"
    GRADIO_SERVER_PORT: int = 7860
    GRADIO_SHARE: bool = False

    @property
    def data_dir(self) -> Path:
        return self.PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def vectorstore_dir(self) -> Path:
        return self.data_dir / "vectorstore"

    @property
    def pdf_file(self) -> Path:
        if self.PDF_PATH:
            return Path(self.PDF_PATH)
        # Default: look in raw directory
        raw = self.raw_dir
        pdfs = list(raw.glob("*.pdf"))
        if pdfs:
            return pdfs[0]
        return raw / "s71200_system_manual_en-US_en-US.pdf"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
