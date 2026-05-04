"""
Task 2: CLI prediction tool using the saved best model.
"""

import argparse
import json
import pathlib
import pickle

import numpy as np
import pandas as pd

BASE_DIR   = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
RESULTS_DIR = BASE_DIR / "results"

FEATURES = ["stride_length_cm", "ground_contact_ms", "hip_drop_degrees", "fatigue_index"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["model_name"]


def predict(model, stride_length_cm: float, ground_contact_ms: float,
            hip_drop_degrees: float, fatigue_index: float) -> float:
    X = pd.DataFrame([{
        "stride_length_cm":  stride_length_cm,
        "ground_contact_ms": ground_contact_ms,
        "hip_drop_degrees":  hip_drop_degrees,
        "fatigue_index":     fatigue_index,
    }])
    return float(model.predict(X)[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Biomotion Injury Risk Predictor")
    parser.add_argument("--stride_length_cm",  type=float, required=True)
    parser.add_argument("--ground_contact_ms", type=float, required=True)
    parser.add_argument("--hip_drop_degrees",  type=float, required=True)
    parser.add_argument("--fatigue_index",     type=float, required=True)
    parser.add_argument("--save_result",       action="store_true",
                        help="Save result to results/step2_s3.json")
    return parser.parse_args()


def main():
    args = parse_args()
    model, model_name = load_model()

    prediction = predict(
        model,
        args.stride_length_cm,
        args.ground_contact_ms,
        args.hip_drop_degrees,
        args.fatigue_index,
    )

    print(f"Predicted injury_risk_score: {prediction:.4f}")

    if args.save_result:
        RESULTS_DIR.mkdir(exist_ok=True)
        output = {
            "image_name": "biomotion-predictor",
            "image_tag": "v1",
            "base_image": "python:3.12-slim",
            "test_input": {
                "stride_length_cm":  args.stride_length_cm,
                "ground_contact_ms": args.ground_contact_ms,
                "hip_drop_degrees":  args.hip_drop_degrees,
                "fatigue_index":     args.fatigue_index,
            },
            "prediction": round(prediction, 4),
        }
        out_path = RESULTS_DIR / "step2_s3.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Result saved: {out_path}")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
