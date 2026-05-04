"""
Task 4: Retraining pipeline — combine old + new data, retrain same model type,
compare champion vs retrained, promote if improvement >= 0.5 RMSE.
"""

import json
import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

BASE_DIR         = pathlib.Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH  = BASE_DIR / "data" / "training_data.csv"
NEW_DATA_PATH    = BASE_DIR / "data" / "new_data.csv"
MODEL_PATH       = BASE_DIR / "models" / "best_model.pkl"
RESULTS_DIR      = BASE_DIR / "results"

FEATURES = ["stride_length_cm", "ground_contact_ms", "hip_drop_degrees", "fatigue_index"]
TARGET   = "injury_risk_score"

RANDOM_STATE = 42
TEST_SIZE    = 0.2
MIN_IMPROVEMENT_THRESHOLD = 0.5


def load_model():
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["model_name"]


def compute_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_model_like(model_name: str):
    """Instantiate a fresh model of the same type as champion."""
    if model_name == "Lasso":
        return Lasso(alpha=1.0, max_iter=1000, random_state=RANDOM_STATE)
    elif model_name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_new   = pd.read_csv(NEW_DATA_PATH)
    df_combined = pd.concat([df_train, df_new], ignore_index=True)

    original_rows = len(df_train)
    new_rows      = len(df_new)
    combined_rows = len(df_combined)

    # ── Split combined data ────────────────────────────────────────────────
    X = df_combined[FEATURES]
    y = df_combined[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── Champion evaluation ────────────────────────────────────────────────
    champion_model, model_name = load_model()
    champion_preds = champion_model.predict(X_test)
    champion_rmse  = compute_rmse(y_test, champion_preds)

    # ── Retrain ────────────────────────────────────────────────────────────
    retrained_model = build_model_like(model_name)
    retrained_model.fit(X_train, y_train)
    retrained_preds = retrained_model.predict(X_test)
    retrained_rmse  = compute_rmse(y_test, retrained_preds)

    # ── Promotion decision ─────────────────────────────────────────────────
    improvement = champion_rmse - retrained_rmse
    action = "promoted" if improvement >= MIN_IMPROVEMENT_THRESHOLD else "kept_champion"

    if action == "promoted":
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": retrained_model, "model_name": model_name}, f)
        print(f"Retrained model promoted and saved to {MODEL_PATH}")
    else:
        print("Champion model retained (improvement below threshold).")

    # ── Save JSON ──────────────────────────────────────────────────────────
    output = {
        "original_data_rows":        original_rows,
        "new_data_rows":             new_rows,
        "combined_data_rows":        combined_rows,
        "champion_rmse":             round(champion_rmse, 4),
        "retrained_rmse":            round(retrained_rmse, 4),
        "improvement":               round(improvement, 4),
        "min_improvement_threshold": MIN_IMPROVEMENT_THRESHOLD,
        "action":                    action,
        "comparison_metric":         "rmse",
    }

    out_path = RESULTS_DIR / "step4_s8.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved: {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
