from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


def load_iris_frame() -> pd.DataFrame:
    dataset = load_iris(as_frame=True)
    frame = dataset.frame.copy()
    frame["species"] = frame["target"].map(dict(enumerate(dataset.target_names)))
    return frame.drop(columns=["target"])


def create_local_iris_csv(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = load_iris_frame()
    frame.to_csv(output, index=False)
    return output