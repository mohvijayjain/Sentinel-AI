import logging
import time

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)

from src.common.error_redaction import describe_error
from src.rag.config import (
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_LLM_MAX_RETRIES,
    NVIDIA_LLM_RETRY_DELAY_SECONDS,
    NVIDIA_LLM_TIMEOUT_SECONDS,
    NVIDIA_MODEL,
    validate_config,
)


logger = logging.getLogger(__name__)

validate_config()

# Explicit, bounded timeout. The SDK's own retries are disabled
# (max_retries=0) so the only retries are the classified, bounded ones in
# generate_answer(); see config.llm_worst_case_seconds().
_client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
    timeout=httpx.Timeout(
        NVIDIA_LLM_TIMEOUT_SECONDS,
        connect=min(5.0, NVIDIA_LLM_TIMEOUT_SECONDS),
    ),
    max_retries=0,
)

# Transient server-side statuses worth one more try. Every other 4xx
# (auth, invalid key, malformed request, unknown model) is permanent.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# The model reasons (hidden reasoning_content) before answering unless told
# not to. For grounded answers over a few short records that deliberation
# added 1,200-1,800 tokens and 39-46s per answer ("why" questions most),
# past the 25s per-attempt timeout, so /chat failed with APITimeoutError.
# With it off the same answers take ~1-2s. Same model, same prompt.
NO_REASONING = {"chat_template_kwargs": {"enable_thinking": False}}


def is_retryable(error: Exception) -> bool:
    """Timeouts, connection failures, 429 and selected 5xx only."""

    # APITimeoutError is a subclass of APIConnectionError
    if isinstance(error, (APIConnectionError, RateLimitError)):
        return True

    if isinstance(error, APIStatusError):
        return error.status_code in RETRYABLE_STATUS_CODES

    return False


def generate_answer(
    question: str,
    context: str,
    *,
    client=None,
    sleep=time.sleep,
) -> str:
    """
    Generate an answer from the retrieved context.

    Bounded: at most 1 + NVIDIA_LLM_MAX_RETRIES attempts, each limited by
    NVIDIA_LLM_TIMEOUT_SECONDS, with NVIDIA_LLM_RETRY_DELAY_SECONDS between
    them; only transient failures are retried. On a permanent failure or
    exhaustion the original exception is raised (/chat maps it to its
    existing 500), and only a redacted description is logged.

    client / sleep are injectable for tests.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context.strip():
        raise ValueError("Context cannot be empty.")

    client = _client if client is None else client
    attempts = 1 + NVIDIA_LLM_MAX_RETRIES

    for attempt in range(1, attempts + 1):

        try:
            return _complete(client, question, context)

        except Exception as error:

            if not is_retryable(error) or attempt == attempts:
                logger.error(
                    "NVIDIA LLM request failed (attempt %d/%d, %s): %s",
                    attempt,
                    attempts,
                    "giving up" if is_retryable(error) else "not retryable",
                    describe_error(error),
                )
                raise

            logger.warning(
                "NVIDIA LLM request failed (attempt %d/%d), retrying in "
                "%.1fs: %s",
                attempt,
                attempts,
                NVIDIA_LLM_RETRY_DELAY_SECONDS,
                describe_error(error),
            )

            sleep(NVIDIA_LLM_RETRY_DELAY_SECONDS)


def _complete(client, question: str, context: str) -> str:

    response = client.chat.completions.create(
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
        extra_body=NO_REASONING,
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