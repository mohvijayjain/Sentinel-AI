import pickle 
import json
import os
from dotenv import load_dotenv
from mlflow_logger import MLflowLogger

load_dotenv()

def log_baseline_model():
    print("Loading model files...")
    
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
        
    with open ("model/metrics.json","r") as f:
        metrics = json.load(f)
        
    with open("model/best_params.json","r") as f:
        params = json.load(f)
    
    with open("model/features.json", "r") as f:
        features = json.load(f)
        
    print(f"  Model loaded")
    print(f"   RMSE : {metrics['rmse']} seconds")
    print(f"   MAE  : {metrics['mae']} seconds")
    print(f"   R²   : {metrics['r2']}")
    
    logger = MLflowLogger(
        experiment_name="Sentinel-AI",
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    
    with logger.start_run(run_name="baseline_champion_v1"):
        logger.log_params(params)
        logger.log_metrics(metrics)
        logger.log_artifact("model/features.json")
        logger.log_artifact("model/feature_importance.csv")
        logger.log_params(
            {
                "model_type": "LightGBM",
                "target": "trip_duration",
                "trained_on": "NYC Taxi 2026 Jan-Mar",
                "n_features": len(features),
                "training_rows": 693380
            }
        )
        
        logger.log_model(model)
        
        logger.log_json(
            {
                "features": features
            },
            "features.json")
        
        import mlflow
        run_id = mlflow.active_run().info.run_id
        print(f"run logged: {run_id}")
        
    version = logger.register_model(
        run_id=run_id,
        model_name = "sentinel-ai-champion"
    )
    logger.set_champions(
        model_name="sentinel-ai-champion",
        model_version=version
    )
    
    print(f"\n🎉 Baseline model successfully logged to MLflow!")
    print(f"Experiment: Sentinel-AI")
    print(f"Run ID: {run_id}")
    print(f"Version: {version}")
    print(f"Stage: Production")
    print(f"\n→ Check: http://localhost:5000")

if __name__ == "__main__":
    log_baseline_model()