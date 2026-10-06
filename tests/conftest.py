"""
Shared test setup for the Sentinel-AI drift scoring suite.

drift_scorer imports, at module top, three modules that drag in live
infrastructure on import (DB engine, NVIDIA/OpenAI embedding client that
raises without NVIDIA_API_KEY, mlflow/lightgbm). We register fake
modules under those names in sys.modules BEFORE anything imports
drift_scorer, so the suite runs with no services, no env vars and no
network. Python's import system returns a cached sys.modules entry
without importing its parent packages, so the real modules never load.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ============================================================
# Make `src.` importable regardless of how pytest is launched
# ============================================================

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ============================================================
# Stub infra-bound modules imported by drift_scorer
# ============================================================

STUBBED_MODULES = {
    "src.database.drift_repository": [
        "insert_monitoring_run",
        "insert_drift_scores",
    ],
    "src.rag.monitoring_updater": [
        "upsert_monitoring_run",
    ],
    "src.training.orchestrator": [
        "run_retraining_pipeline",
    ],
}


def _install_stubs():

    for module_name, attributes in STUBBED_MODULES.items():

        stub = types.ModuleType(module_name)
        stub.__file__ = f"<stub {module_name}>"

        for attribute in attributes:
            setattr(stub, attribute, MagicMock(name=attribute))

        sys.modules[module_name] = stub


# Runs at conftest import time, i.e. before any test module is collected
_install_stubs()


# ============================================================
# CSV headers, matching the real files in reports/
# ============================================================

STATISTICAL_COLUMNS = [
    "feature", "type", "psi", "ks_statistic", "ks_pvalue",
    "js_divergence", "wasserstein", "severity",
]

SHAP_COLUMNS = [
    "feature", "ref_importance", "cur_importance", "absolute_shift",
    "relative_shift_%", "direction", "ref_rank", "cur_rank",
    "rank_shift", "is_drifted", "severity",
]

PREDICTION_COLUMNS = [
    "psi", "ks_statistic", "ks_pvalue", "js_divergence",
    "wasserstein", "severity",
]


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def drift_scorer():
    """The drift_scorer module, imported against the stubs."""

    from src.monitoring import drift_scorer as module

    return module


@pytest.fixture
def write_report(tmp_path):
    """
    Write a report CSV with the real header into tmp_path.

    Usage: write_report("statistical", [{"severity": "LOW"}, ...])
    Columns not given in a row are filled with neutral placeholders.
    """

    columns_by_kind = {
        "statistical": STATISTICAL_COLUMNS,
        "shap": SHAP_COLUMNS,
        "prediction": PREDICTION_COLUMNS,
    }

    def _write(kind, rows):

        columns = columns_by_kind[kind]
        defaults = {column: 0 for column in columns}
        defaults.update(feature="f", type="numerical", direction="INCREASED")

        records = [
            {column: {**defaults, **row}.get(column) for column in columns}
            for row in rows
        ]

        path = tmp_path / f"{kind}_drift.csv"

        pd.DataFrame(records, columns=columns).to_csv(path, index=False)

        return str(path)

    return _write
