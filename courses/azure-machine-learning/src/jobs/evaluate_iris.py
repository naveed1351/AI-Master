from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model_path)
    frame = pd.read_csv(args.test_data)
    features = frame.drop(columns=["species"])
    labels = frame["species"]
    predictions = model.predict(features)

    payload = {
        "accuracy": accuracy_score(labels, predictions),
        "report": classification_report(labels, predictions, output_dict=True),
    }
    output_path = Path("outputs") / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()