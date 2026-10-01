from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "telco-churn-xgboost"
SELECTION_METRIC = "test_roc_auc"
