from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, default="")
    parser.add_argument("--model-output", type=str, default="outputs")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=200)
    return parser.parse_args()


def load_training_frame(train_data: str) -> tuple[pd.DataFrame, pd.Series]:
    if train_data:
        frame = pd.read_csv(train_data)
        features = frame.drop(columns=["species"])
        labels = frame["species"]
        return features, labels

    dataset = load_iris(as_frame=True)
    features = dataset.data
    labels = dataset.target_names[dataset.target]
    return features, pd.Series(labels, name="species")


def main() -> None:
    args = parse_args()
    features, labels = load_training_frame(args.train_data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_metric("accuracy", accuracy)

        output_dir = Path(args.model_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_dir / "model.joblib")
        metrics = {
            "accuracy": accuracy,
            "n_estimators": args.n_estimators,
            "rows": len(features),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics))


if __name__ == "__main__":
    main()