# src/rag/rag.py

"""
Complete Sentinel-AI RAG pipeline.

Flow:
    User Question
        ↓
    Retriever
        ↓
    Relevant ChromaDB chunks
        ↓
    Context
        ↓
    LLM
        ↓
    Final Answer
"""

from typing import List, Dict, Any

from src.rag.config import TOP_K
from src.rag.retriever import retrieve
from src.rag.llm import generate_answer


# ============================================================
# Context Builder
# ============================================================

def build_context(
    results: List[Dict[str, Any]],
) -> str:
    """
    Convert retrieved documents into LLM context.
    """

    if not results:
        return ""

    context_parts = []

    for i, result in enumerate(results, start=1):

        document = result.get("document", "")

        if document:
            context_parts.append(
                f"[Source {i}]\n{document}"
            )

    return "\n\n".join(context_parts)


# ============================================================
# RAG Pipeline
# ============================================================

def ask(
    question: str,
    top_k: int = TOP_K,
) -> str:
    """
    Execute the complete RAG pipeline.

    User question
        → retrieval
        → context construction
        → LLM generation
    """

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    # --------------------------------------------------------
    # 1. Retrieve relevant knowledge
    # --------------------------------------------------------

    results = retrieve(
        query=question,
        top_k=top_k,
    )

    # --------------------------------------------------------
    # 2. Build context
    # --------------------------------------------------------

    context = build_context(results)

    if not context:
        return (
            "The requested information is not available "
            "in the Sentinel-AI knowledge base."
        )

    # --------------------------------------------------------
    # 3. Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        question=question,
        context=context,
    )

    return answer


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":

    questions = [
        "What is the current drift status?",
        "वर्तमान ड्रिफ्ट स्थिति क्या है?",
        "Quel est l'état actuel du drift ?",
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("=" * 70)

        answer = ask(question)

        print("\nANSWER:")
        print(answer)