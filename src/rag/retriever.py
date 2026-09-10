
# src/rag/retriever.py

from typing import Any, Dict, List

from src.rag.config import TOP_K
from src.rag.embeddings import embed_query
from src.rag.chroma_store import ChromaStore


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
    ) -> List[Dict[str, Any]]:
        """
        Convert the user query into an embedding and
        retrieve the most relevant documents.

        Returned items are ordered best-first. Each carries a
        `distance` (see note below), not a similarity score.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query = query.strip()

        # ----------------------------------------------------
        # Generate query embedding
        #
        # Must use embed_query (input_type="query"). Nemotron is
        # asymmetric: encoding a query as a passage silently
        # degrades retrieval quality.
        # ----------------------------------------------------

        query_embedding = embed_query(query)

        # ----------------------------------------------------
        # Search ChromaDB
        # ----------------------------------------------------

        results = self.store.query(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        # ----------------------------------------------------
        # Convert ChromaDB response into clean results
        #
        # Chroma nests each field one level deep (one list per
        # query). We only ever send a single query, so we take
        # index [0], defaulting to [[]] to stay safe on empty
        # results.
        # ----------------------------------------------------

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        ids = results.get(
            "ids",
            [[]],
        )[0]

        retrieved = []

        for i, document in enumerate(documents):

            retrieved.append(
                {
                    "id": ids[i] if i < len(ids) else None,
                    "document": document,
                    # NOTE: Chroma returns a DISTANCE, not a
                    # similarity. For a cosine collection this is
                    # (1 - cosine_similarity), so SMALLER is more
                    # relevant. Convert/invert before any code
                    # that expects "higher = better".
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
# Convenience Function
# ============================================================

_retriever = None


def get_retriever() -> RAGRetriever:
    """
    Return a reusable RAG retriever instance.

    The instance caches TOP_K from first construction. To tune
    top_k interactively, build a fresh RAGRetriever(top_k=...)
    instead of using this singleton.
    """

    global _retriever

    if _retriever is None:
        _retriever = RAGRetriever()

    return _retriever


def retrieve(
    query: str,
) -> List[Dict[str, Any]]:
    """
    Convenience function for retrieving relevant documents.
    """

    return get_retriever().retrieve(query)

