from src.rag.embeddings import embed_documents, embed_text
from src.rag.chroma_store import ChromaStore


# ============================================================
# Test documents
# ============================================================

documents = [
    "The challenger model was rejected because it failed the RMSE evaluation gate.",
    "The champion model achieved an RMSE of 326.56 and an R2 score of 0.8628.",
    "The challenger model achieved an RMSE of 347.63 and an R2 score of 0.8446.",
]


# ============================================================
# Generate embeddings
# ============================================================

print("Generating document embeddings...")

embeddings = embed_documents(documents)

print(f"Generated {len(embeddings)} embeddings.")
print(f"Embedding dimensions: {len(embeddings[0])}")


# ============================================================
# Initialize ChromaDB
# ============================================================

store = ChromaStore()

print(f"Existing documents: {store.count()}")


# ============================================================
# Add documents
# ============================================================

ids = [
    "test_doc_1",
    "test_doc_2",
    "test_doc_3",
]

metadatas = [
    {"type": "evaluation"},
    {"type": "champion"},
    {"type": "challenger"},
]

store.add_documents(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas,
)

print(f"Documents after insertion: {store.count()}")


# ============================================================
# Test English query
# ============================================================

hindi_query = "Challenger model reject kyu hua?"

print("\n\nHindi / Hinglish Query:")
print(hindi_query)

hindi_embedding = embed_text(hindi_query)

hindi_results = store.query(
    query_embedding=hindi_embedding,
    top_k=3,
)


print("\nRetrieved documents for Hindi / Hinglish query:")

for i, document in enumerate(hindi_results["documents"][0]):
    distance = hindi_results["distances"][0][i]

    print(f"\nResult {i + 1}")
    print(f"Distance: {distance}")
    print(f"Document: {document}")