# src/rag/ingest.py

"""
Sentinel-AI RAG ingestion pipeline.

Flow:
    documents.py
        ↓
    chunker.py
        ↓
    embeddings.py
        ↓
    ChromaDB
"""

from typing import List, Dict, Any

from src.rag.documents import load_project_documents
from src.rag.chunker import chunk_text
from src.rag.embeddings import embed_passages
from src.rag.chroma_store import ChromaStore


def ingest() -> None:
    """Build/update the Sentinel-AI RAG knowledge base."""

    # --------------------------------------------------------
    # 1. Load project documents
    # --------------------------------------------------------

    documents = load_project_documents()

    if not documents:
        print("No documents found.")
        return

    print(f"Loaded {len(documents)} source documents.")

    # --------------------------------------------------------
    # 2. Chunk documents
    # --------------------------------------------------------

    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for document in documents:

        document_chunks = chunk_text(
            document["text"]
        )

        for chunk in document_chunks:

            chunks.append(chunk)

            metadatas.append(
                document["metadata"]
            )

    if not chunks:
        print("No valid chunks generated.")
        return

    print(f"Generated {len(chunks)} chunks.")

    # --------------------------------------------------------
    # 3. Generate passage embeddings
    # --------------------------------------------------------

    print("Generating embeddings...")

    embeddings = embed_passages(chunks)

    print(f"Generated {len(embeddings)} embeddings.")

    # --------------------------------------------------------
    # 4. Store in ChromaDB
    # --------------------------------------------------------

    store = ChromaStore()

    # IDs must be unique
    ids = [
        f"sentinel_chunk_{i}"
        for i in range(len(chunks))
    ]

    store.add_documents(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    print(
        f"Stored {len(chunks)} chunks in ChromaDB."
    )

    print(
        f"Total documents in collection: {store.count()}"
    )


if __name__ == "__main__":
    ingest()