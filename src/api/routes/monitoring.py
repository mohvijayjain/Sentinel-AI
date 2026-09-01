from fastapi import APIRouter

from src.database.drift_repository import (
    get_latest_monitoring_run,
    get_monitoring_history,
    get_drifted_features,
    get_feature_drift_scores
)


router = APIRouter(
    prefix="/monitoring",
    tags=["Monitoring"]
)


@router.get("/latest")
def latest_monitoring():

    result = get_latest_monitoring_run()

    if result is None:
        return {
            "message": "No monitoring data found"
        }

    return dict(result._mapping)



@router.get("/history")
def monitoring_history(
    limit: int = 20
):

    results = get_monitoring_history(
        limit
    )

    return [
        dict(row._mapping)
        for row in results
    ]



@router.get("/drifted-features")
def drifted_features():

    results = get_drifted_features()

    return [
        dict(row._mapping)
        for row in results
    ]



@router.get("/feature-scores")
def feature_scores():

    results = get_feature_drift_scores()

    return [
        dict(row._mapping)
        for row in results
    ]