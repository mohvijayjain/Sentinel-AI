"""
src/rag/embeddings.py

Embedding generation using NVIDIA's hosted embedding API
through the OpenAI-compatible endpoint.

Model:
    nvidia/nemotron-3-embed-1b

Properties:
    - 2048-dimensional dense vectors
    - Multilingual retrieval
    - Supports query/passage asymmetric embeddings
    - Query and passage inputs use different input_type values

Usage:
    Documents/chunks:
        embed_passages(["document 1", "document 2"])

    User query:
        embed_query("Why was the challenger rejected?")

Note:
    embed() is strictly 1:1. It does NOT drop empty inputs; it
    raises on them, so returned vectors always align with the
    input list. Filter and validate chunks during chunking,
    before they reach this module.

Run the self-test from the project root:
    python -m src.rag.embeddings
"""

import time
from typing import List, Literal

from openai import OpenAI

from src.rag.config import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    validate_config,
)


# ============================================================
# Configuration
# ============================================================

MAX_BATCH = 64
DEFAULT_MAX_RETRIES = 3

InputType = Literal["query", "passage"]


# ============================================================
# Configuration Validation
# ============================================================

validate_config()


# ============================================================
# NVIDIA Client
# ============================================================

_client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
)


# ============================================================
# Internal Batch Embedding
# ============================================================

def _embed_batch(
    texts: List[str],
    input_type: InputType,
) -> List[List[float]]:
    """
    Generate embeddings for one batch.

    input_type:
        - "query"   -> user/search queries
        - "passage" -> documents/chunks stored in ChromaDB

    The distinction is important because Nemotron uses
    asymmetric retrieval.
    """

    if input_type not in ("query", "passage"):
        raise ValueError(
            "input_type must be either 'query' or 'passage'."
        )

    if not texts:
        return []

    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
        extra_body={
            "input_type": input_type,
            "truncate": "END",
        },
    )

    # API normally returns results in input order.
    # Sorting by index guarantees the original order.
    data = sorted(
        response.data,
        key=lambda item: item.index,
    )

    vectors = [
        item.embedding
        for item in data
    ]

    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    if len(vectors) != len(texts):
        raise RuntimeError(
            f"NVIDIA returned {len(vectors)} embeddings "
            f"for {len(texts)} inputs."
        )

    for i, vector in enumerate(vectors):

        if len(vector) != EMBEDDING_DIM:
            raise RuntimeError(
                f"Unexpected embedding dimension at index {i}: "
                f"got {len(vector)}, "
                f"expected {EMBEDDING_DIM}."
            )

    return vectors


# ============================================================
# Input Validation
# ============================================================

def _validate_texts(texts: List[str]) -> List[str]:
    """
    Enforce a strict 1:1 contract.

    Every element must be a non-empty string. Whitespace is
    trimmed, but an input that is empty or whitespace-only is
    an error, not something to silently drop. This keeps the
    returned vectors aligned with the caller's input list
    (and, downstream, with parallel chunk IDs and metadata).
    """

    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings.")

    cleaned: List[str] = []

    for i, text in enumerate(texts):

        if not isinstance(text, str):
            raise TypeError(
                f"texts[{i}] is not a string: {type(text).__name__}."
            )

        stripped = text.strip()

        if not stripped:
            raise ValueError(
                f"texts[{i}] is empty or whitespace-only. "
                f"Filter empty chunks before embedding."
            )

        cleaned.append(stripped)

    return cleaned


# ============================================================
# Public Embedding Function
# ============================================================

def embed(
    texts: List[str],
    input_type: InputType,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.

    Strictly 1:1: len(result) == len(texts). Raises on any
    empty or non-string input rather than dropping it.

    Parameters
    ----------
    texts:
        List of non-empty strings to embed.

    input_type:
        "passage" for documents/chunks.
        "query" for user/search queries.

    max_retries:
        Number of retries after an API failure.

    Returns
    -------
    List[List[float]]
        List of embedding vectors, aligned with `texts`.
    """

    if input_type not in ("query", "passage"):
        raise ValueError(
            "input_type must be either 'query' or 'passage'."
        )

    if max_retries < 0:
        raise ValueError(
            "max_retries cannot be negative."
        )

    if not texts:
        return []

    # Validate; raises on empty/non-string, never resizes.
    cleaned_texts = _validate_texts(texts)

    vectors: List[List[float]] = []

    # --------------------------------------------------------
    # Process in batches
    # --------------------------------------------------------

    for start in range(
        0,
        len(cleaned_texts),
        MAX_BATCH,
    ):

        batch = cleaned_texts[
            start:start + MAX_BATCH
        ]

        # ----------------------------------------------------
        # Retry API request
        # ----------------------------------------------------

        for attempt in range(
            max_retries + 1
        ):

            try:

                batch_vectors = _embed_batch(
                    batch,
                    input_type,
                )

                vectors.extend(
                    batch_vectors
                )

                break

            except Exception as exc:

                # Last attempt -> raise the actual error
                if attempt >= max_retries:
                    raise

                wait_seconds = 2 ** attempt

                print(
                    f"NVIDIA embedding request failed "
                    f"(attempt {attempt + 1}/{max_retries + 1}). "
                    f"Retrying in {wait_seconds}s..."
                )

                time.sleep(wait_seconds)

    # Final alignment guarantee.
    if len(vectors) != len(cleaned_texts):
        raise RuntimeError(
            f"Alignment error: produced {len(vectors)} vectors "
            f"for {len(cleaned_texts)} inputs."
        )

    return vectors


# ============================================================
# Passage Embeddings
# ============================================================

def embed_passages(
    texts: List[str],
) -> List[List[float]]:
    """
    Generate embeddings for documents/chunks.

    These vectors are intended to be stored in ChromaDB.
    Returns exactly one vector per input, in order.
    """

    return embed(
        texts,
        input_type="passage",
    )


# ============================================================
# Query Embedding
# ============================================================

def embed_query(
    text: str,
) -> List[float]:
    """
    Generate an embedding for a user/search query.

    Uses input_type="query" because Nemotron uses
    asymmetric query/passage retrieval.
    """

    if not isinstance(text, str):
        raise TypeError(
            "Query must be a string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Query cannot be empty."
        )

    vectors = embed(
        [text],
        input_type="query",
    )

    return vectors[0]


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":

    import math

    def cosine_similarity(
        a: List[float],
        b: List[float],
    ) -> float:

        dot = sum(
            x * y
            for x, y in zip(a, b)
        )

        norm_a = math.sqrt(
            sum(x * x for x in a)
        )

        norm_b = math.sqrt(
            sum(y * y for y in b)
        )

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (
            norm_a * norm_b
        )

    # --------------------------------------------------------
    # Test passages
    # --------------------------------------------------------

    passages = [
        "The challenger model was rejected because it failed the RMSE evaluation gate.",
        "The champion model achieved an RMSE of 326.56 and an R2 score of 0.8628.",
        "The challenger model achieved an RMSE of 347.63 and an R2 score of 0.8446.",
        "\u091a\u0948\u0932\u0947\u0902\u091c\u0930 \u092e\u0949\u0921\u0932 RMSE evaluation gate \u092e\u0947\u0902 fail \u0939\u094b\u0928\u0947 \u0915\u0947 \u0915\u093e\u0930\u0923 reject \u0939\u0941\u0906\u0964",
    ]

    # --------------------------------------------------------
    # Test queries
    # --------------------------------------------------------

    queries = [
        "Why was the challenger model rejected?",
        "Challenger model reject kyu hua?",
        "\u091a\u0948\u0932\u0947\u0902\u091c\u0930 \u092e\u0949\u0921\u0932 \u0915\u094d\u092f\u094b\u0902 \u0930\u093f\u091c\u0947\u0915\u094d\u091f \u0939\u0941\u0906?",
    ]

    print(
        f"Embedding {len(passages)} passages..."
    )

    passage_vectors = embed_passages(
        passages
    )

    print(
        f"Generated {len(passage_vectors)} vectors."
    )

    print(
        f"Embedding dimension: "
        f"{len(passage_vectors[0])}"
    )

    print(
        f"Expected dimension: "
        f"{EMBEDDING_DIM}"
    )

    # --------------------------------------------------------
    # Cross-language retrieval sanity check
    # --------------------------------------------------------

    for query in queries:

        query_vector = embed_query(
            query
        )

        ranked = sorted(
            (
                (
                    cosine_similarity(
                        query_vector,
                        passage_vector,
                    ),
                    passage,
                )
                for passage, passage_vector
                in zip(
                    passages,
                    passage_vectors,
                )
            ),
            reverse=True,
        )

        print("\n" + "=" * 70)

        print(
            f"Query: {query}"
        )

        print("=" * 70)

        for score, passage in ranked[:2]:

            print(
                f"{score:.4f}  {passage}"
            )