from src.database.drift_repository import insert_drift_event


insert_drift_event(
    drifted_features=[
        "pickup_month"
    ],
    action="RETRAIN",
    report_path="reports/drift_summary.json"
)


print("Test completed")