"""Gradio app serving the best churn model from the MLflow experiment."""
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from best_model import load_best_model
from config import SELECTION_METRIC
from data import ID_COLUMN, NUMERIC_FEATURES, RAW_FEATURES, category_options, clean, encode

model, run = load_best_model()
FEATURE_COLUMNS = model.get_booster().feature_names
OPTIONS = category_options()


def score(df):
    X = encode(clean(df), FEATURE_COLUMNS)
    return model.predict_proba(X)[:, 1].astype(float)


def predict_single(*values):
    record = pd.DataFrame([dict(zip(RAW_FEATURES, values))])
    proba = float(score(record)[0])
    label = "Likely to churn" if proba >= 0.5 else "Likely to stay"
    return {label: proba if proba >= 0.5 else 1 - proba}, round(proba, 4)


def predict_batch(csv_file):
    df = pd.read_csv(csv_file)
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise gr.Error(f"CSV is missing columns: {', '.join(missing)}")
    proba = score(df)
    df["churn_probability"] = proba.round(4)
    df["prediction"] = ["Churn" if p >= 0.5 else "No churn" for p in proba]
    front = [c for c in [ID_COLUMN, "prediction", "churn_probability"] if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]
    out_path = Path(tempfile.mkdtemp()) / "predictions.csv"
    df.to_csv(out_path, index=False)
    return df, str(out_path)


def numeric_input(col):
    if col == "SeniorCitizen":
        return gr.Radio([0, 1], value=0, label=col)
    if col == "tenure":
        return gr.Slider(0, 72, value=12, step=1, label="tenure (months)")
    return gr.Number(value={"MonthlyCharges": 70.0, "TotalCharges": 840.0}[col], label=col)


def single_inputs(per_row=3):
    components = []
    for i in range(0, len(RAW_FEATURES), per_row):
        with gr.Row():
            for col in RAW_FEATURES[i : i + per_row]:
                if col in NUMERIC_FEATURES:
                    components.append(numeric_input(col))
                else:
                    components.append(gr.Dropdown(OPTIONS[col], value=OPTIONS[col][0], label=col))
    return components


with gr.Blocks(title="Telco Churn Predictor") as demo:
    gr.Markdown(
        f"# Telco Customer Churn Predictor\n"
        f"Serving MLflow run **{run['tags.mlflow.runName']}** "
        f"(`{run.run_id[:8]}`), selected by {SELECTION_METRIC} = "
        f"{run[f'metrics.{SELECTION_METRIC}']:.4f}."
    )

    with gr.Tab("Single Record"):
        with gr.Row():
            with gr.Column(scale=3):
                inputs = single_inputs()
            with gr.Column(scale=1):
                label_out = gr.Label(label="Prediction")
                proba_out = gr.Number(label="Churn probability")
                gr.Button("Predict", variant="primary").click(
                    predict_single, inputs=inputs, outputs=[label_out, proba_out]
                )

    with gr.Tab("Batch CSV"):
        gr.Markdown("Upload a CSV with the raw dataset columns (see `data/sample_batch.csv`).")
        file_in = gr.File(label="CSV file", file_types=[".csv"], type="filepath")
        table_out = gr.Dataframe(label="Predictions")
        file_out = gr.File(label="Download predictions")
        gr.Button("Score file", variant="primary").click(
            predict_batch, inputs=file_in, outputs=[table_out, file_out]
        )


if __name__ == "__main__":
    demo.launch()
