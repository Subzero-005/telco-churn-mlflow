"""Find the best MLflow run and load its model."""
import mlflow

from config import EXPERIMENT_NAME, SELECTION_METRIC, TRACKING_URI


def best_run():
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{SELECTION_METRIC} DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No finished runs found. Run `python src/train.py` first.")
    return runs.iloc[0]


def load_best_model():
    run = best_run()
    model = mlflow.xgboost.load_model(f"runs:/{run.run_id}/model")
    return model, run


if __name__ == "__main__":
    run = best_run()
    print(f"Best run: {run['tags.mlflow.runName']} ({run.run_id})")
    print(f"{SELECTION_METRIC}: {run[f'metrics.{SELECTION_METRIC}']:.4f}")
