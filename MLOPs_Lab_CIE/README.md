# Biomotion Injury Risk — MLOps Lab

A complete MLOps pipeline that trains machine learning models to predict injury risk scores from biomechanical data, exposes predictions via a REST API, and supports automated retraining.

---

## Project Structure

```
Internals_Basics/
└── MLOPs_Lab_CIE/
    ├── data/
    │   ├── training_data.csv       # Original training data (25 rows)
    │   └── new_data.csv            # New data for retraining (20 rows)
    ├── src/
    │   ├── train.py                # Task 1: Train models + MLflow logging
    │   ├── predict_cli.py          # Task 2: CLI prediction tool
    │   ├── api.py                  # Task 3: FastAPI REST service
    │   └── retrain.py              # Task 4: Retraining pipeline
    ├── models/
    │   └── best_model.pkl          # Saved best model (generated)
    ├── results/
    │   ├── step1_s1.json           # Training results (generated)
    │   ├── step2_s3.json           # CLI prediction result (generated)
    │   ├── step3_s4.json           # API prediction result (generated)
    │   └── step4_s8.json           # Retraining result (generated)
    ├── Dockerfile
    ├── requirements.txt
    ├── .gitignore
    └── README.md
```

---

## Features

| Feature | Details |
|---|---|
| Models | Lasso, GradientBoostingRegressor |
| Experiment Tracking | MLflow (local file store) |
| Best Model Selection | Lowest RMSE |
| REST API | FastAPI on port 8888 |
| CLI Tool | argparse-based predictor |
| Retraining | Champion vs challenger comparison |
| Containerisation | Docker |

---

## Prerequisites

Make sure you have the following installed:

- **Python 3.11+** — https://www.python.org/downloads/
- **pip** — comes with Python
- **Docker Desktop** (optional, for containerisation) — https://www.docker.com/products/docker-desktop/
- **Postman** (optional, for API testing) — https://www.postman.com/downloads/

---

## Setup

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/Internals_Basics.git
cd Internals_Basics/MLOPs_Lab_CIE
```

Or if you downloaded the zip, extract it and navigate into the folder:

```bash
cd Internals_Basics/MLOPs_Lab_CIE
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows users:** If you get a permissions error, try:
> ```bash
> pip install --user -r requirements.txt
> ```

---

## Running the Project

Run each task in order. **Task 1 must be run first** as all other tasks depend on the saved model.

---

### Task 1 — Train Models + MLflow

Trains Lasso and GradientBoosting models, logs to MLflow, selects the best model by RMSE, and saves it.

```bash
python src/train.py
```

**Output:**
- `models/best_model.pkl` — saved best model
- `results/step1_s1.json` — training metrics for both models
- `mlruns/` — MLflow experiment tracking data

**Example result:**
```json
{
  "experiment_name": "biomotion-injury-risk-score",
  "models": [
    {
      "name": "Lasso",
      "mae": 3.242,
      "rmse": 4.2128,
      "r2": 0.9263
    },
    {
      "name": "GradientBoosting",
      "mae": 9.1544,
      "rmse": 10.2404,
      "r2": 0.5645
    }
  ],
  "best_model": "Lasso",
  "best_metric_name": "rmse",
  "best_metric_value": 4.2128
}
```

---

### Task 2 — CLI Prediction

Runs a single prediction from the command line.

**Windows (PowerShell) — all on one line:**
```powershell
python src/predict_cli.py --stride_length_cm 108.6 --ground_contact_ms 252.1 --hip_drop_degrees 6.7 --fatigue_index 4.2 --save_result
```

**Windows (PowerShell) — multi-line with backtick:**
```powershell
python src/predict_cli.py `
  --stride_length_cm 108.6 `
  --ground_contact_ms 252.1 `
  --hip_drop_degrees 6.7 `
  --fatigue_index 4.2 `
  --save_result
```

**Mac/Linux — multi-line with backslash:**
```bash
python src/predict_cli.py \
  --stride_length_cm 108.6 \
  --ground_contact_ms 252.1 \
  --hip_drop_degrees 6.7 \
  --fatigue_index 4.2 \
  --save_result
```

**Output:**
- Prints predicted `injury_risk_score` to terminal
- `results/step2_s3.json` — saved prediction result

> **Note:** The `--save_result` flag is optional. Without it, the prediction is still printed but not saved to a file.

---

### Task 3 — FastAPI REST Service

Starts the API server on port 8888.

```bash
python src/api.py
```

The terminal will show:
```
INFO:     Uvicorn running on http://0.0.0.0:8888 (Press CTRL+C to quit)
```

This is normal — the server is running and waiting for requests. **Leave this terminal open.**

#### Testing with Postman

**GET /heartbeat**
| Field | Value |
|---|---|
| Method | `GET` |
| URL | `http://localhost:8888/heartbeat` |

Expected response:
```json
{
  "status": "running",
  "model": "Lasso",
  "version": "1.0"
}
```

---

**POST /forecast**
| Field | Value |
|---|---|
| Method | `POST` |
| URL | `http://localhost:8888/forecast` |
| Body type | `raw` → `JSON` |

Request body:
```json
{
  "stride_length_cm": 108.6,
  "ground_contact_ms": 252.1,
  "hip_drop_degrees": 6.7,
  "fatigue_index": 4.2
}
```

Expected response:
```json
{
  "injury_risk_score": 38.8038
}
```

---

**POST /forecast — Invalid input (should return 422)**

```json
{
  "stride_length_cm": 999,
  "ground_contact_ms": 252.1,
  "hip_drop_degrees": 6.7,
  "fatigue_index": 4.2
}
```

Expected response: **HTTP 422 Unprocessable Entity**

#### Input Validation Rules

| Field | Min | Max |
|---|---|---|
| stride_length_cm | 80 | 160 |
| ground_contact_ms | 150 | 350 |
| hip_drop_degrees | 2 | 15 |
| fatigue_index | 1 | 10 |

#### Interactive API Docs

With the server running, open your browser and visit:
```
http://localhost:8888/docs
```
This opens the Swagger UI where you can test all endpoints visually.

**Output:**
- `results/step3_s4.json` — saved on first POST /forecast call

**Stop the server:** Press `CTRL+C` in the terminal when done.

---

### Task 4 — Retraining Pipeline

Combines old and new data, retrains the same model type, and compares against the champion model.

```bash
python src/retrain.py
```

**Output:**
- `results/step4_s8.json` — comparison results and promotion decision

**Example result:**
```json
{
  "original_data_rows": 25,
  "new_data_rows": 20,
  "combined_data_rows": 45,
  "champion_rmse": 1.4813,
  "retrained_rmse": 1.3991,
  "improvement": 0.0822,
  "min_improvement_threshold": 0.5,
  "action": "kept_champion",
  "comparison_metric": "rmse"
}
```

> If `improvement >= 0.5`, the retrained model is promoted and `best_model.pkl` is updated. Otherwise the champion is kept.

---

## Docker (Optional)

Build and run the CLI predictor inside a Docker container.

### Prerequisites
- Docker Desktop must be running

### Step 1 — Build the image

Run this from inside `MLOPs_Lab_CIE/` (where the Dockerfile lives):

```bash
docker build -t biomotion-predictor:v1 .
```

> **Important:** Run `python src/train.py` first so `models/best_model.pkl` exists before building. The model is baked into the image.

### Step 2 — Run the container

**Windows (PowerShell) — one line:**
```powershell
docker run biomotion-predictor:v1 --stride_length_cm 108.6 --ground_contact_ms 252.1 --hip_drop_degrees 6.7 --fatigue_index 4.2
```

**Mac/Linux:**
```bash
docker run biomotion-predictor:v1 \
  --stride_length_cm 108.6 \
  --ground_contact_ms 252.1 \
  --hip_drop_degrees 6.7 \
  --fatigue_index 4.2
```

### Step 3 — Save image to file (optional)

```bash
docker save biomotion-predictor:v1 -o biomotion-predictor-v1.tar
```

### Step 4 — Load image from file (optional)

```bash
docker load -i biomotion-predictor-v1.tar
```

---

## MLflow UI (Optional)

View experiment runs in a browser dashboard.

```bash
mlflow ui
```

Then open: **http://localhost:5000**

---

## Common Errors & Fixes

### `ModuleNotFoundError`
Dependencies not installed. Run:
```bash
pip install -r requirements.txt
```

### MLflow `file://` URI error on Windows
This is a Windows path issue. Make sure `train.py` uses `.as_uri()` for the tracking URI:
```python
mlruns_path = BASE_DIR / "mlruns"
mlruns_path.mkdir(exist_ok=True)
mlflow.set_tracking_uri(mlruns_path.as_uri())
```

### `FileNotFoundError: best_model.pkl`
You must run `train.py` before any other script:
```bash
python src/train.py
```

### PowerShell `--` syntax error
PowerShell does not support `\` for line continuation. Use backtick `` ` `` instead, or put everything on one line.

### Port 8888 already in use
Another process is using the port. Either stop that process or change the port in `api.py`:
```python
uvicorn.run("api:app", host="0.0.0.0", port=8889, reload=False)
```

### `UserWarning: X does not have valid feature names`
This is just a warning, not an error — predictions still work correctly. To suppress it, pass a pandas DataFrame instead of a numpy array in `api.py`:
```python
import pandas as pd
X = pd.DataFrame([{
    "stride_length_cm":  data.stride_length_cm,
    "ground_contact_ms": data.ground_contact_ms,
    "hip_drop_degrees":  data.hip_drop_degrees,
    "fatigue_index":     data.fatigue_index,
}])
```

---

## Full Run Order (Quick Reference)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train
python src/train.py

# 3. CLI predict
python src/predict_cli.py --stride_length_cm 108.6 --ground_contact_ms 252.1 --hip_drop_degrees 6.7 --fatigue_index 4.2 --save_result

# 4. Start API (leave running, open new terminal for next steps)
python src/api.py

# 5. Retrain (new terminal)
python src/retrain.py

# 6. Docker (optional)
docker build -t biomotion-predictor:v1 .
docker run biomotion-predictor:v1 --stride_length_cm 108.6 --ground_contact_ms 252.1 --hip_drop_degrees 6.7 --fatigue_index 4.2
```

---

## Data Schema

| Column | Description | Unit |
|---|---|---|
| `stride_length_cm` | Length of one stride | cm |
| `ground_contact_ms` | Time foot is on ground | milliseconds |
| `hip_drop_degrees` | Hip drop angle | degrees |
| `fatigue_index` | Fatigue level score | 1–10 |
| `injury_risk_score` | **Target** — predicted injury risk | score |
