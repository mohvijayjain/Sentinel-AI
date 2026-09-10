# src/rag/config.py

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"

CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"


# ============================================================
# ChromaDB
# ============================================================

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "sentinel_knowledge",
)


# ============================================================
# LLM — NVIDIA API
# ============================================================

NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY"
)

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "deepseek-ai/deepseek-v4-pro-0813",
)


# ============================================================
# Embeddings
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nvidia/llama-nemotron-embed-1b-v2",
)

# ============================================================
# Chunking
# ============================================================

CHUNK_SIZE = int(
    os.getenv(
        "RAG_CHUNK_SIZE",
        "800",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "RAG_CHUNK_OVERLAP",
        "120",
    )
)


# ============================================================
# Retrieval
# ============================================================

TOP_K = int(
    os.getenv(
        "RAG_TOP_K",
        "5",
    )
)


# ============================================================
# Validation
# ============================================================

def validate_config() -> None:
    """
    Validate required Sentinel-AI RAG configuration.
    """

    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. "
            "Add it to the .env file."
        )

    if CHUNK_SIZE <= 0:
        raise ValueError(
            "RAG_CHUNK_SIZE must be greater than 0."
        )

    if CHUNK_OVERLAP < 0:
        raise ValueError(
            "RAG_CHUNK_OVERLAP cannot be negative."
        )

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "RAG_CHUNK_OVERLAP must be smaller "
            "than RAG_CHUNK_SIZE."
        )

    if TOP_K <= 0:
        raise ValueError(
            "RAG_TOP_K must be greater than 0."
        )