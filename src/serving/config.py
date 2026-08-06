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
API_KEY = os.getenv("API_KEY", "sentinel-secret")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")