from openai import OpenAI

from src.rag.config import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    validate_config,
)


validate_config()

_client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
)


def generate_answer(
    question: str,
    context: str,
) -> str:
    """Generate an answer using DeepSeek with retrieved context."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context.strip():
        raise ValueError("Context cannot be empty.")

    response = _client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Sentinel-AI's AI assistant. "
                    "Answer using only the provided context. "
                    "If the context does not contain the answer, "
                    "say that the information is unavailable. "
                    "Respond in the same language as the user's question. "
                    "If the user explicitly requests a different language, "
                    "respond in that language."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":

    answer = generate_answer(
        question="वर्तमान ड्रिफ्ट स्थिति क्या है?",
        context=(
            'Sentinel-AI drift monitoring report: '
            '{"statistical_score": 0, '
            '"shap_score": 0, '
            '"prediction_score": 0, '
            '"overall_score": 0.0, '
            '"action": "WAIT"}.'
        ),
    )

    print("\nAnswer:")
    print(answer)