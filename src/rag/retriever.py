# src/rag/retriever.py

"""
Retrieval layer for Sentinel-AI RAG.

Responsibilities:
    1. Convert user query into an embedding.
    2. Search ChromaDB for relevant documents.
    3. Return clean, structured retrieval results.
"""

from typing import Any, Dict, List

from src.rag.config import TOP_K
from src.rag.embeddings import embed_query
from src.rag.chroma_store import ChromaStore


# ============================================================
# RAG Retriever
# ============================================================

class RAGRetriever:
    """
    Handles query embedding and similarity retrieval
    from the Sentinel-AI ChromaDB knowledge base.
    """

    def __init__(
        self,
        top_k: int = TOP_K,
    ):
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        self.top_k = top_k
        self.store = ChromaStore()

    # ========================================================
    # Retrieve
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert the user query into an embedding and
        retrieve the most relevant documents.

        top_k can be specified per query.
        """

        # ----------------------------------------------------
        # Validate query
        # ----------------------------------------------------

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query = query.strip()

        # ----------------------------------------------------
        # Validate top_k
        # ----------------------------------------------------

        k = self.top_k if top_k is None else top_k

        if k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        # ----------------------------------------------------
        # Generate query embedding
        # ----------------------------------------------------

        query_embedding = embed_query(query)

        # ----------------------------------------------------
        # Search ChromaDB
        # ----------------------------------------------------

        results = self.store.query(
            query_embedding=query_embedding,
            top_k=k,
        )

        # ----------------------------------------------------
        # Extract ChromaDB results
        # ----------------------------------------------------

        documents = results.get(
            "documents",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        ids = results.get(
            "ids",
            [[]],
        )[0]

        # ----------------------------------------------------
        # Build clean result objects
        # ----------------------------------------------------

        retrieved: List[Dict[str, Any]] = []

        for i, document in enumerate(documents):

            retrieved.append(
                {
                    "id": (
                        ids[i]
                        if i < len(ids)
                        else None
                    ),
                    "document": document,
                    "distance": (
                        distances[i]
                        if i < len(distances)
                        else None
                    ),
                    "metadata": (
                        metadatas[i]
                        if i < len(metadatas)
                        else {}
                    ),
                }
            )

        return retrieved


# ============================================================
# Reusable Retriever Instance
# ============================================================

_retriever = None


def get_retriever() -> RAGRetriever:
    """
    Return a reusable RAG retriever instance.
    """

    global _retriever

    if _retriever is None:
        _retriever = RAGRetriever()

    return _retriever


# ============================================================
# Convenience Function
# ============================================================

def retrieve(
    query: str,
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function for retrieving relevant documents.
    """

    return get_retriever().retrieve(
        query=query,
        top_k=top_k,
    )