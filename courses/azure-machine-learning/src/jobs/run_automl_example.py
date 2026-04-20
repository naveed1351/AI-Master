from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from azureml_course.dataset_utils import create_local_iris_csv
from azureml_course.training_jobs import build_automl_classification_job
from azureml_course.workspace import get_ml_client


def main() -> None:
    client = get_ml_client()
    data_path = create_local_iris_csv(Path("data") / "iris.csv")
    automl_job = build_automl_classification_job(str(data_path.resolve()))
    created_job = client.jobs.create_or_update(automl_job)
    print(f"Submitted AutoML job: {created_job.name}")


if __name__ == "__main__":
    main()