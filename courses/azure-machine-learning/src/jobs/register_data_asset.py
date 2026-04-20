from __future__ import annotations

from pathlib import Path
import sys

from azure.ai.ml.entities import Data

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from azureml_course.dataset_utils import create_local_iris_csv
from azureml_course.workspace import get_ml_client


def main() -> None:
    client = get_ml_client()
    dataset_path = create_local_iris_csv(Path("data") / "iris.csv")
    data_asset = Data(
        name="iris-course-data",
        path=str(dataset_path.resolve()),
        type="uri_file",
        description="Iris dataset for the Azure ML course",
        version="1",
    )
    client.data.create_or_update(data_asset)
    print(f"Registered data asset from {dataset_path}")


if __name__ == "__main__":
    main()