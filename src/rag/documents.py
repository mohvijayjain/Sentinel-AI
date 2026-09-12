# src/rag/documents.py

"""
Document loader for Sentinel-AI RAG.

Reads project outputs and converts them into
RAG-ready documents with metadata.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from src.rag.config import PROJECT_ROOT


# ============================================================
# Helpers
# ============================================================

def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Model Metrics
# ============================================================

def load_model_metrics() -> List[Dict[str, Any]]:
    """
    Load model metrics and convert them into
    RAG-ready documents.
    """

    path = PROJECT_ROOT / "model" / "metrics.json"

    data = _load_json(path)

    if not data:
        return []

    documents = []

    for model_name, metrics in data.items():

        if not isinstance(metrics, dict):
            continue

        text = (
            f"Model: {model_name}. "
            f"Model metrics: "
            f"{json.dumps(metrics, ensure_ascii=False)}."
        )

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": "model/metrics.json",
                    "type": "model_metrics",
                    "model": model_name,
                },
            }
        )

    return documents


# ============================================================
# Drift Summary
# ============================================================

def load_drift_summary() -> List[Dict[str, Any]]:
    """
    Load drift monitoring results.
    """

    path = PROJECT_ROOT / "reports" / "drift_summary.json"

    data = _load_json(path)

    if not data:
        return []

    text = (
        "Sentinel-AI drift monitoring report: "
        f"{json.dumps(data, ensure_ascii=False)}."
    )

    return [
        {
            "text": text,
            "metadata": {
                "source": "reports/drift_summary.json",
                "type": "drift_report",
            },
        }
    ]


# ============================================================
# Load All Project Documents
# ============================================================

def load_project_documents() -> List[Dict[str, Any]]:
    """
    Load all currently supported Sentinel-AI knowledge sources.
    """

    documents: List[Dict[str, Any]] = []

    documents.extend(load_model_metrics())
    documents.extend(load_drift_summary())

    return documents


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":

    documents = load_project_documents()

    print("=" * 70)
    print("SENTINEL-AI DOCUMENT LOADER TEST")
    print("=" * 70)

    print(f"Documents loaded: {len(documents)}")

    for i, document in enumerate(documents, start=1):

        print(f"\n--- Document {i} ---")
        print("Type:", document["metadata"]["type"])
        print("Source:", document["metadata"]["source"])
        print("Text:", document["text"])