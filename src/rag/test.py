from src.rag.retriever import retrieve


queries = [
    "What is the current drift status?",
    "Drift status kya hai?",
    "ड्रिफ्ट की स्थिति क्या है?",
]


for query in queries:

    print("\n" + "=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    results = retrieve(query)

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Distance:", result["distance"])
        print("Document:", result["document"])
        print("Metadata:", result["metadata"])