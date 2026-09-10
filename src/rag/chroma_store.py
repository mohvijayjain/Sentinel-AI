# src/rag/chroma_store.py

from typing import Any, Dict, List, Optional

import chromadb

from src.rag.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
)


class ChromaStore:
    """
    Persistent ChromaDB store for Sentinel-AI RAG.
    """

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        self.collection_name = collection_name

        # Create persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        # Get existing collection or create a new one
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    # ========================================================
    # Add Documents
    # ========================================================

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents and their embeddings to ChromaDB.
        """

        if not documents:
            return

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match "
                "number of embeddings."
            )

        if len(documents) != len(ids):
            raise ValueError(
                "Number of documents must match "
                "number of IDs."
            )

        if metadatas is not None:
            if len(documents) != len(metadatas):
                raise ValueError(
                    "Number of documents must match "
                    "number of metadata entries."
                )

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ========================================================
    # Query
    # ========================================================

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrieve the most relevant documents.
        """

        if not query_embedding:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    # ========================================================
    # Collection Info
    # ========================================================

    def count(self) -> int:
        """
        Return number of stored documents.
        """

        return self.collection.count()

    # ========================================================
    # Get Documents
    # ========================================================

    def get(
        self,
        ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve stored documents.

        If ids is None, retrieve all documents.
        """

        if ids:
            return self.collection.get(
                ids=ids
            )

        return self.collection.get()

    # ========================================================
    # Delete Documents
    # ========================================================

    def delete(
        self,
        ids: List[str],
    ) -> None:
        """
        Delete documents by ID.
        """

        if not ids:
            return

        self.collection.delete(
            ids=ids
        )

    # ========================================================
    # Clear Collection
    # ========================================================

    def clear(self) -> None:
        """
        Delete all documents from the collection.
        """

        existing = self.collection.get()

        ids = existing.get("ids", [])

        if ids:
            self.collection.delete(
                ids=ids
            )