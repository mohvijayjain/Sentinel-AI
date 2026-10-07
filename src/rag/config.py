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
# NVIDIA API (shared endpoint for LLM + embeddings)
# ============================================================

NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY"
)

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)


# ============================================================
# LLM — NVIDIA API
# ============================================================

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
)

# Bounded LLM latency for one /chat answer. Worst case:
#     timeout * (1 + max_retries) + max_retries * retry_delay
#   = 25 * (1 + 1) + 1 * 2 = 52s with the defaults below,
# and validate_config() rejects any override above the 60s ceiling.
# Without this the OpenAI SDK defaults (600s timeout, 2 retries) applied.
NVIDIA_LLM_TIMEOUT_SECONDS = float(
    os.getenv(
        "NVIDIA_LLM_TIMEOUT_SECONDS",
        "25",
    )
)

NVIDIA_LLM_MAX_RETRIES = int(
    os.getenv(
        "NVIDIA_LLM_MAX_RETRIES",
        "1",
    )
)

NVIDIA_LLM_RETRY_DELAY_SECONDS = float(
    os.getenv(
        "NVIDIA_LLM_RETRY_DELAY_SECONDS",
        "2",
    )
)

LLM_LATENCY_CEILING_SECONDS = 60.0


def llm_worst_case_seconds(
    timeout: float = None,
    max_retries: int = None,
    retry_delay: float = None,
) -> float:
    """Worst-case total LLM time for one answer, all attempts included."""

    timeout = NVIDIA_LLM_TIMEOUT_SECONDS if timeout is None else timeout
    max_retries = NVIDIA_LLM_MAX_RETRIES if max_retries is None else max_retries
    retry_delay = (
        NVIDIA_LLM_RETRY_DELAY_SECONDS if retry_delay is None else retry_delay
    )

    return timeout * (1 + max_retries) + max_retries * retry_delay


# ============================================================
# Embeddings — NVIDIA
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nvidia/nemotron-3-embed-1b",
)

EMBEDDING_DIM = int(
    os.getenv(
        "EMBEDDING_DIM",
        "2048",
    )
)


# ============================================================
# Chunking
# ============================================================

CHUNK_SIZE = int(
    os.getenv(
        "RAG_CHUNK_SIZE",
        "1000",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "RAG_CHUNK_OVERLAP",
        "300",
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

# Relevance cutoff on Chroma's COSINE DISTANCE (the collection is created
# with hnsw:space=cosine): distance = 1 - cosine_similarity, range 0..2,
# SMALLER = MORE relevant. Retrieved documents with distance ABOVE this
# value are dropped before the LLM sees them; if none remain, /chat
# returns the "information unavailable" answer without calling the LLM.
#
# Default 0.85 (cosine similarity >= 0.15), calibrated on the live index
# with nvidia/nemotron-3-embed-1b: relevant hits for real monitoring /
# retraining questions in English, Hindi and French were 0.35-0.72, while
# the closest document to off-topic questions (capital of France, recipes,
# weather, ...) was 0.93-1.04. 0.85 sits in that gap, nearer the junk
# side, so real questions keep a ~0.13 margin. Raise it (max 2.0 = no
# filtering) if legitimate questions start coming back "unavailable".
RAG_MAX_DISTANCE = float(
    os.getenv(
        "RAG_MAX_DISTANCE",
        "0.85",
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

    if not NVIDIA_BASE_URL:
        raise RuntimeError(
            "NVIDIA_BASE_URL is not configured."
        )

    if not EMBEDDING_MODEL:
        raise RuntimeError(
            "EMBEDDING_MODEL is not configured."
        )

    if EMBEDDING_DIM <= 0:
        raise ValueError(
            "EMBEDDING_DIM must be greater than 0."
        )

    if not NVIDIA_MODEL:
        raise RuntimeError(
            "NVIDIA_MODEL is not configured."
        )

    if NVIDIA_LLM_TIMEOUT_SECONDS <= 0:
        raise ValueError(
            "NVIDIA_LLM_TIMEOUT_SECONDS must be greater than 0."
        )

    if NVIDIA_LLM_MAX_RETRIES < 0 or NVIDIA_LLM_RETRY_DELAY_SECONDS < 0:
        raise ValueError(
            "NVIDIA_LLM_MAX_RETRIES and NVIDIA_LLM_RETRY_DELAY_SECONDS "
            "cannot be negative."
        )

    if llm_worst_case_seconds() > LLM_LATENCY_CEILING_SECONDS:
        raise ValueError(
            f"LLM worst-case latency {llm_worst_case_seconds():.0f}s exceeds "
            f"the {LLM_LATENCY_CEILING_SECONDS:.0f}s ceiling: lower "
            "NVIDIA_LLM_TIMEOUT_SECONDS, NVIDIA_LLM_MAX_RETRIES or "
            "NVIDIA_LLM_RETRY_DELAY_SECONDS."
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

    if not 0 < RAG_MAX_DISTANCE <= 2:
        raise ValueError(
            "RAG_MAX_DISTANCE must be in (0, 2] (cosine distance)."
        )