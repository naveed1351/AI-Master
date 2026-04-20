from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AzureMLSettings:
    subscription_id: str
    resource_group: str
    workspace_name: str
    compute_name: str = "cpu-cluster"
    datastore_name: str = "workspaceblobstore"
    online_endpoint_name: str = "azureml-course-endpoint"
    deployment_name: str = "blue"
    location: str = "eastus"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]


def get_settings() -> AzureMLSettings:
    return AzureMLSettings(
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID", ""),
        resource_group=os.getenv("AZURE_RESOURCE_GROUP", ""),
        workspace_name=os.getenv("AZURE_ML_WORKSPACE", ""),
        compute_name=os.getenv("AZURE_ML_COMPUTE", "cpu-cluster"),
        datastore_name=os.getenv("AZURE_ML_DATASTORE", "workspaceblobstore"),
        online_endpoint_name=os.getenv("AZURE_ML_ONLINE_ENDPOINT", "azureml-course-endpoint"),
        deployment_name=os.getenv("AZURE_ML_DEPLOYMENT", "blue"),
        location=os.getenv("AZURE_LOCATION", "eastus"),
    )


def validate_settings(settings: AzureMLSettings) -> list[str]:
    missing: list[str] = []
    if not settings.subscription_id:
        missing.append("AZURE_SUBSCRIPTION_ID")
    if not settings.resource_group:
        missing.append("AZURE_RESOURCE_GROUP")
    if not settings.workspace_name:
        missing.append("AZURE_ML_WORKSPACE")
    return missing