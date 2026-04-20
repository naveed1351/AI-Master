from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from azureml_course.workspace import get_ml_client, workspace_summary


def main() -> None:
    client = get_ml_client()
    summary = workspace_summary()
    workspace = client.workspaces.get(summary["workspace_name"])
    print(f"Connected to workspace: {workspace.name}")
    print(f"Location: {workspace.location}")


if __name__ == "__main__":
    main()