# Drift scoring test suite

Tests for Sentinel-AI's drift severity and scoring logic.

## Running

```bash
pytest tests/
```

You don't need any environment variables or running services: no Postgres, ChromaDB, NVIDIA API key, network access or mlflow. Run `pytest tests/`, not a bare `pytest`, so the repository-level `test_repository.py` (which needs a live database) isn't collected.

## What's covered

| File | Target |
|---|---|
| `test_severity_score.py` | `drift_scorer.severity_score`: every known label, plus the warning path for unknown labels (including the `"DRIFT"` and `"MEDIUM_shift"` regressions) |
| `test_get_action.py` | `drift_scorer.get_action`: every action band |
| `test_calculate_scores.py` | `calculate_statistical_score`, `calculate_shap_score` and `calculate_prediction_score`, run against temp CSVs with the real report headers |
| `test_detectors.py` | `stastical_drift.get_psi_severity` (shared by the statistical and prediction layers), and the severity bands and `is_drifted` triggers in `shap_drift.analyze_shap_drift` |
| `test_overall_bands.py` | The 0.4 / 0.3 / 0.3 weighting contract mapped to actions, both directly and end to end from CSVs |

Every threshold written as `<` is tested on both sides. The exact cutoff value must land in the upper band, and a value just below it (for example `0.249999`) must land in the lower band.

## Why `conftest.py` stubs modules

At import time, `src/monitoring/drift_scorer.py` imports:

- `src.database.drift_repository`, which creates a DB engine
- `src.rag.monitoring_updater`, which builds the NVIDIA embedding client and raises if `NVIDIA_API_KEY` is missing
- `src.training.orchestrator`, which pulls in mlflow and lightgbm

Before any test module is collected, `conftest.py` registers fake modules under those three names in `sys.modules`, each exposing `MagicMock` versions of the imported functions. Python returns a module that's already in `sys.modules` without importing it or its parent packages, so none of the real infrastructure code runs. `test_severity_score.py::test_infra_modules_are_stubbed` checks that this still holds.

`conftest.py` also puts the repository root on `sys.path`, so `src.` imports resolve however pytest is launched.

The scorer reads the module-level constants `STATISTICAL_PATH`, `SHAP_PATH` and `PREDICTION_PATH`. Tests monkeypatch these to point at per-test CSVs in `tmp_path`, so the real `reports/` directory is never read or written.

## Scope

This suite only adds tests. It makes no changes to production code.
