"""Train XGBoost churn classifiers across a small hyperparameter sweep, tracked with MLflow autologging."""
import mlflow
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import EXPERIMENT_NAME, TRACKING_URI
from data import load_train_test

PARAM_GRID = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03},
    {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
    {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.01},
]


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.xgboost.autolog()

    X_train, X_test, y_train, y_test = load_train_test()

    for params in PARAM_GRID:
        run_name = "xgb_n{n_estimators}_d{max_depth}_lr{learning_rate}".format(**params)
        with mlflow.start_run(run_name=run_name):
            model = xgb.XGBClassifier(
                **params,
                subsample=0.9,
                colsample_bytree=0.8,
                eval_metric=["logloss", "auc"],
                random_state=42,
            )
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            proba = model.predict_proba(X_test)[:, 1]
            preds = (proba >= 0.5).astype(int)
            metrics = {
                "test_accuracy": accuracy_score(y_test, preds),
                "test_precision": precision_score(y_test, preds),
                "test_recall": recall_score(y_test, preds),
                "test_f1": f1_score(y_test, preds),
                "test_roc_auc": roc_auc_score(y_test, proba),
            }
            mlflow.log_metrics(metrics)
            print(f"{run_name:28s} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
