from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd


MODEL = None


def init() -> None:
    global MODEL
    model_root = Path(os.getenv("AZUREML_MODEL_DIR", Path(__file__).resolve().parent))
    model_paths = list(model_root.rglob("model.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"No model.joblib found under {model_root}")
    MODEL = joblib.load(model_paths[0])


def run(raw_data: str) -> str:
    data = json.loads(raw_data)
    frame = pd.DataFrame(data)
    predictions = MODEL.predict(frame)
    return json.dumps({"predictions": predictions.tolist()})