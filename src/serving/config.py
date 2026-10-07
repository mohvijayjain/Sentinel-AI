import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model Paths
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "model/features.json")
METRICS_PATH = os.getenv("METRICS_PATH", "model/metrics.json")
BEST_PARAMS_PATH = os.getenv("BEST_PARAMS_PATH", "model/best_params.json")

# API Configuration
#
# No default: a guessable built-in key would silently authenticate a
# misconfigured deployment. Unset/blank -> None, and the app refuses to
# start (require_api_key() in lifespan); the routes also reject every
# request while it is None.
API_KEY = (os.getenv("API_KEY") or "").strip() or None


def require_api_key() -> str:
    """Return the configured API key or fail fast with a clear error."""

    if not API_KEY:
        raise RuntimeError(
            "API_KEY is not set. Set a strong, unique API_KEY in the "
            "environment / .env before starting the Sentinel-AI API."
        )

    return API_KEY
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# CORS
#
# Explicit allowlist, comma-separated. Unset/blank -> local dev frontend
# only; never "*": a wildcard with allow_credentials=True is invalid per
# the CORS spec, and an open API would be callable from any web page.
DEFAULT_CORS_ALLOWED_ORIGINS = ["http://localhost:3000"]


def parse_cors_origins(raw):
    """Split a comma-separated origin list: strip, drop empties, refuse '*'."""

    origins = [
        origin.strip().rstrip("/")
        for origin in (raw or "").split(",")
        if origin.strip()
    ]

    if "*" in origins:
        raise RuntimeError(
            "CORS_ALLOWED_ORIGINS must list explicit origins; '*' is not "
            "allowed."
        )

    return origins or list(DEFAULT_CORS_ALLOWED_ORIGINS)


CORS_ALLOWED_ORIGINS = parse_cors_origins(os.getenv("CORS_ALLOWED_ORIGINS"))
