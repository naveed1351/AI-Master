from __future__ import annotations

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from .config import AzureMLSettings, get_settings, validate_settings


def get_ml_client(settings: AzureMLSettings | None = None) -> MLClient:
    settings = settings or get_settings()
    missing = validate_settings(settings)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required Azure ML settings: {joined}")

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    return MLClient(
        credential=credential,
        subscription_id=settings.subscription_id,
        resource_group_name=settings.resource_group,
        workspace_name=settings.workspace_name,
    )


def workspace_summary(settings: AzureMLSettings | None = None) -> dict[str, str]:
    settings = settings or get_settings()
    return {
        "subscription_id": settings.subscription_id,
        "resource_group": settings.resource_group,
        "workspace_name": settings.workspace_name,
        "compute_name": settings.compute_name,
        "datastore_name": settings.datastore_name,
        "location": settings.location,
    }