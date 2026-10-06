"""
IGNORED_FEATURES lives in the dependency-free src/monitoring/constants.py
and both the SHAP detector and the scorer use that one list.

Import-side checks run in a fresh interpreter: this test session has
already loaded shap via the detector tests, so sys.modules here proves
nothing.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from src.monitoring import constants


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _run_isolated(code):
    """Run `code` in a fresh interpreter rooted at the repo."""

    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr

    return completed.stdout


# ============================================================
# One authoritative list
# ============================================================

def test_ignored_features_unchanged():

    assert constants.IGNORED_FEATURES == ["pickup_month"]


def test_shap_detector_uses_constants_list():

    # The detector module itself needs shap; skip rather than error
    pytest.importorskip("shap")

    from src.monitoring import shap_drift

    assert shap_drift.IGNORED_FEATURES is constants.IGNORED_FEATURES


def test_scorer_uses_constants_list(drift_scorer):

    assert drift_scorer.SHAP_IGNORED_FEATURES is constants.IGNORED_FEATURES


# ============================================================
# Import side effects
# ============================================================

def test_constants_module_imports_nothing_heavy():

    stdout = _run_isolated("""
        import sys
        before = set(sys.modules)
        import src.monitoring.constants
        added = set(sys.modules) - before
        heavy = {"shap", "pandas", "numpy", "scipy", "lightgbm", "mlflow"}
        print(sorted(m for m in added if m.split(".")[0] in heavy))
    """)

    assert stdout.strip() == "[]"


def test_scorer_imports_without_shap():
    """
    With shap made unimportable, drift_scorer must still import, and
    neither shap nor the SHAP detector module may be loaded by it.
    DB / RAG / retraining modules are stubbed exactly as in conftest.
    """

    stdout = _run_isolated(f"""
        import sys
        sys.modules["shap"] = None          # any `import shap` now fails
        sys.path.insert(0, {TESTS_DIR!r})
        import conftest                     # installs the infra stubs
        import src.monitoring.drift_scorer
        print(sys.modules.get("shap"),
              "src.monitoring.shap_drift" in sys.modules)
    """)

    assert stdout.strip() == "None False"
