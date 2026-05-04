"""
Task 3: FastAPI REST service with /heartbeat and /forecast endpoints.
Runs on port 8888.
"""

import json
import pathlib
import pickle

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

BASE_DIR    = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / "models" / "best_model.pkl"
RESULTS_DIR = BASE_DIR / "results"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["model_name"]


MODEL, MODEL_NAME = load_model()

app = FastAPI(title="Biomotion Injury Risk API", version="1.0")


# ── Schemas ────────────────────────────────────────────────────────────────

class BiomotionInput(BaseModel):
    stride_length_cm:  float = Field(..., ge=80,  le=160)
    ground_contact_ms: float = Field(..., ge=150, le=350)
    hip_drop_degrees:  float = Field(..., ge=2,   le=15)
    fatigue_index:     float = Field(..., ge=1,   le=10)


class HeartbeatResponse(BaseModel):
    status:  str
    model:   str
    version: str


class ForecastResponse(BaseModel):
    injury_risk_score: float


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/heartbeat", response_model=HeartbeatResponse)
def heartbeat():
    return {"status": "running", "model": MODEL_NAME, "version": "1.0"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(data: BiomotionInput):
    X = pd.DataFrame([{
        "stride_length_cm":  data.stride_length_cm,
        "ground_contact_ms": data.ground_contact_ms,
        "hip_drop_degrees":  data.hip_drop_degrees,
        "fatigue_index":     data.fatigue_index,
    }])
    prediction = float(MODEL.predict(X)[0])

    # Save step3 result on first forecast call
    RESULTS_DIR.mkdir(exist_ok=True)
    result_path = RESULTS_DIR / "step3_s4.json"
    output = {
        "heartbeat_endpoint": "/heartbeat",
        "forecast_endpoint":  "/forecast",
        "port":               8888,
        "model_name":         MODEL_NAME,
        "test_input": {
            "stride_length_cm":  data.stride_length_cm,
            "ground_contact_ms": data.ground_contact_ms,
            "hip_drop_degrees":  data.hip_drop_degrees,
            "fatigue_index":     data.fatigue_index,
        },
        "prediction": round(prediction, 4),
    }
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2)

    return {"injury_risk_score": round(prediction, 4)}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8888, reload=False)
