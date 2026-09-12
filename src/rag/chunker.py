"""
src/rag/chunker.py

Text chunking layer for Sentinel-AI RAG.

Responsibilities:
    1. Clean raw text.
    2. Split text into overlapping chunks.
    3. Return only valid, non-empty chunks.

This module does NOT:
    - generate embeddings
    - access ChromaDB
    - call the LLM
    - perform retrieval
"""

import re
from typing import List

from src.rag.config import CHUNK_SIZE, CHUNK_OVERLAP


# ============================================================
# Text Cleaning
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean raw document text.

    Removes excessive whitespace while preserving
    meaningful text structure.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    text = text.replace("\x00", " ")

    # Normalize repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# ============================================================
# Chunk Validation
# ============================================================

def _validate_chunk_config(
    chunk_size: int,
    chunk_overlap: int,
) -> None:

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )


# ============================================================
# Text Chunking
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Parameters
    ----------
    text:
        Raw document text.

    chunk_size:
        Maximum number of characters per chunk.

    chunk_overlap:
        Number of characters shared between consecutive chunks.

    Returns
    -------
    List[str]
        Clean, non-empty text chunks.
    """

    _validate_chunk_config(
        chunk_size,
        chunk_overlap,
    )

    text = clean_text(text)

    if not text:
        return []

    chunks: List[str] = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


# ============================================================
# Multiple Documents
# ============================================================

def chunk_documents(
    documents: List[str],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Chunk multiple documents into a single list of chunks.
    """

    if not isinstance(documents, list):
        raise TypeError(
            "documents must be a list of strings."
        )

    chunks: List[str] = []

    for document in documents:

        document_chunks = chunk_text(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks.extend(document_chunks)

    return chunks


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":

    sample_text = """
    Sentinel-AI is an ML Model Monitoring and Retraining Platform.

    It monitors model performance, detects drift, evaluates
    challenger models, and manages model promotion.

    The RAG layer allows users to ask questions about the
    Sentinel-AI system using natural language.
    """

    chunks = chunk_text(sample_text)

    print("=" * 70)
    print("CHUNKER TEST")
    print("=" * 70)

    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):

        print(f"\n--- Chunk {i} ---")
        print(chunk)
        print(f"Characters: {len(chunk)}")