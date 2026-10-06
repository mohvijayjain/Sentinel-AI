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