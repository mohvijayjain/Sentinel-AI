"""
Deterministic secret redaction for error text that gets persisted.

A failed retraining attempt's error message is stored in PostgreSQL and
embedded into ChromaDB, where RAG can surface it, so anything that looks
like a credential is replaced with REDACTED before either store sees it.

Dependency-free (re only) so the orchestrator, the repository and the
RAG layer can all share it.

Redacted:
  * passwords in URLs              scheme://user:<secret>@host
  * Authorization header values    Authorization: <scheme> <secret>
  * key=value / key: value pairs whose key names a password, secret,
    token, API/access/private key or credential (NVIDIA_API_KEY=...,
    password=..., client_secret: "...", Pwd=...;)
  * Bearer tokens                  Bearer <token containing a digit>
  * well-known key formats         sk-..., nvapi-..., AKIA..., ghp_...,
                                   xox?-..., JWTs (eyJ...)

Kept on purpose: hostnames, ports, paths, user names, numbers and
ordinary words (e.g. "password authentication failed", "max_tokens=5"),
which are needed to debug a failure.

safe_error_message() is idempotent, including on its own truncated
output: values starting with "[" or "." are never treated as secrets, so
REDACTED, a truncated "[REDAC..." and the "..." marker are stable.
"""

import re
import traceback


REDACTED = "[REDACTED]"

# Matches the CHECK on retraining_events.error_message
ERROR_MESSAGE_MAX_LENGTH = 500

_TRUNCATION_MARKER = "..."

# A secret value never starts with "[" or "." (keeps output stable)
_VALUE_START = r"(?![\[.])"

_UNQUOTED_VALUE = r"[^\s,;&'\"\[\](){}<>]+"

_QUOTED_VALUE = r"\"[^\"]{2,}\"|'[^']{2,}'"

_SECRET_KEY = (
    r"\b(?:[a-z0-9]+[_\-.])*"
    r"(?:password|passwd|pwd|secret|token|api[_\-]?key|apikey"
    r"|access[_\-]?key|private[_\-]?key|client[_\-]?secret"
    r"|auth[_\-]?token|credentials?)\b"
)

_PATTERNS = [
    # scheme://user:password@host -> scheme://user:[REDACTED]@host
    (
        re.compile(
            r"(?P<keep>\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:)[^\s@/]+(?=@)",
            re.IGNORECASE,
        ),
        r"\g<keep>" + REDACTED,
    ),
    # Authorization: Bearer abc / "Authorization": "Basic abc"
    (
        re.compile(
            r"(?P<keep>\bauthorization\b[\"']?\s*[:=]\s*[\"']?)"
            r"(?:(?:bearer|basic|token|digest)\s+)?"
            + _VALUE_START + _UNQUOTED_VALUE,
            re.IGNORECASE,
        ),
        r"\g<keep>" + REDACTED,
    ),
    # password=..., NVIDIA_API_KEY=..., "client_secret": "...", Pwd=...;
    (
        re.compile(
            r"(?P<keep>" + _SECRET_KEY + r"[\"']?\s*[:=]\s*)"
            + _VALUE_START + r"(?:" + _QUOTED_VALUE + r"|" + _UNQUOTED_VALUE + r")",
            re.IGNORECASE,
        ),
        r"\g<keep>" + REDACTED,
    ),
    # Bearer <token>; requires a digit so "Bearer authentication" stays
    (
        re.compile(
            r"(?P<keep>\bbearer\s+)" + _VALUE_START
            + r"(?=[A-Za-z0-9\-._~+/]*[0-9])[A-Za-z0-9\-._~+/]{8,}=*",
            re.IGNORECASE,
        ),
        r"\g<keep>" + REDACTED,
    ),
    # Well-known key formats, wherever they appear
    (
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_\-]{16,}"
            r"|nvapi-[A-Za-z0-9_\-]{16,}"
            r"|AKIA[0-9A-Z]{16}"
            r"|gh[pousr]_[A-Za-z0-9]{20,}"
            r"|xox[abprs]-[A-Za-z0-9\-]{10,}"
            r"|eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+)"
        ),
        REDACTED,
    ),
]


def redact_secrets(text: str) -> str:
    """Replace credential-like substrings with REDACTED."""

    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)

    return text


def safe_error_message(
    text: str,
    max_length: int = ERROR_MESSAGE_MAX_LENGTH,
) -> str:
    """
    Make error text safe to persist: one line, secrets redacted, then
    bounded. Redaction runs BEFORE truncation so a cut can never leave
    half a secret behind.
    """

    message = redact_secrets(" ".join(str(text).split()))

    if len(message) > max_length:
        message = (
            message[:max_length - len(_TRUNCATION_MARKER)]
            + _TRUNCATION_MARKER
        )

    return message


def describe_error(error: BaseException) -> str:
    """'<Type>: <message>', one line, redacted and bounded (for print)."""

    return safe_error_message(f"{type(error).__name__}: {error}")


def redacted_traceback(error: BaseException) -> str:
    """
    The full formatted traceback (chained causes included) with secrets
    redacted, for logs. Multi-line and unbounded on purpose: it replaces
    logger.exception() output, not the persisted error_message. The
    exception object itself is not modified.
    """

    formatted = "".join(
        traceback.format_exception(
            type(error), error, error.__traceback__
        )
    )

    return redact_secrets(formatted).rstrip()
