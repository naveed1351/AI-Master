from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from azureml_course.deployment_samples import build_online_endpoint
from azureml_course.workspace import get_ml_client


def main() -> None:
    client = get_ml_client()
    endpoint = build_online_endpoint()
    created_endpoint = client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Created endpoint: {created_endpoint.name}")


if __name__ == "__main__":
    main()