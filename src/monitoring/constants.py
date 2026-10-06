"""
Dependency-free monitoring constants.

Kept free of heavy imports (shap, pandas, ...) so the scorer can share
them with the detectors without loading detector dependencies.
"""

# Features the SHAP detector and the drift scorer never treat as drift
IGNORED_FEATURES=[
    "pickup_month"
]
