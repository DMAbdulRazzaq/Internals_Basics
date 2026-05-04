"""
Task 1: Train Lasso and GradientBoostingRegressor models, log to MLflow,
select best model by RMSE, save model and results.
"""

import os
import json
import pickle
import pathlib

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "training_data.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

FEATURES = ["stride_length_cm", "ground_contact_ms", "hip_drop_degrees", "fatigue_index"]
TARGET   = "injury_risk_score"

EXPERIMENT_NAME = "biomotion-injury-risk-score"
RANDOM_STATE    = 42
TEST_SIZE       = 0.2


def load_data(path: pathlib.Path):
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def compute_metrics(y_true, y_pred) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_log(name: str, model, X_train, X_test, y_train, y_test, params: dict) -> dict:
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("experiment_type", "baseline_comparison")
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)

        mlflow.log_metrics({"mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"]})
        mlflow.sklearn.log_model(model, artifact_path=name)

    return {"name": name, **metrics, "model_obj": model}


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── MLflow setup ───────────────────────────────────────────────────────
    mlruns_path = BASE_DIR / "mlruns"
    mlruns_path.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(mlruns_path.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Data ───────────────────────────────────────────────────────────────
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── Model definitions ──────────────────────────────────────────────────
    lasso_params = {"alpha": 1.0, "max_iter": 1000, "random_state": RANDOM_STATE}
    gb_params    = {
        "n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
        "random_state": RANDOM_STATE
    }

    lasso_model = Lasso(**lasso_params)
    gb_model    = GradientBoostingRegressor(**gb_params)

    # ── Train & log ────────────────────────────────────────────────────────
    lasso_result = train_and_log("Lasso",             lasso_model, X_train, X_test, y_train, y_test, lasso_params)
    gb_result    = train_and_log("GradientBoosting",  gb_model,    X_train, X_test, y_train, y_test, gb_params)

    results = [lasso_result, gb_result]

    # ── Select best model ──────────────────────────────────────────────────
    best = min(results, key=lambda r: r["rmse"])

    # ── Save best model ────────────────────────────────────────────────────
    model_path = MODEL_DIR / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best["model_obj"], "model_name": best["name"]}, f)
    print(f"Best model saved: {model_path}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    output = {
        "experiment_name": EXPERIMENT_NAME,
        "models": [
            {"name": r["name"], "mae": round(r["mae"], 4), "rmse": round(r["rmse"], 4), "r2": round(r["r2"], 4)}
            for r in results
        ],
        "best_model": best["name"],
        "best_metric_name": "rmse",
        "best_metric_value": round(best["rmse"], 4),
    }

    out_path = RESULTS_DIR / "step1_s1.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved: {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
