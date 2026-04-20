from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from azureml_course.pipelines import build_training_pipeline
from azureml_course.workspace import get_ml_client


def main() -> None:
    client = get_ml_client()
    pipeline_job = build_training_pipeline()
    created_job = client.jobs.create_or_update(pipeline_job)
    print(f"Submitted pipeline job: {created_job.name}")


if __name__ == "__main__":
    main()