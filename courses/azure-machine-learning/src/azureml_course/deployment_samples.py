from __future__ import annotations

from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)

from .config import AzureMLSettings, get_settings


def build_online_endpoint(settings: AzureMLSettings | None = None) -> ManagedOnlineEndpoint:
    settings = settings or get_settings()
    return ManagedOnlineEndpoint(
        name=settings.online_endpoint_name,
        description="Managed online endpoint for the Azure ML course",
        auth_mode="key",
        tags={"course": "azure-machine-learning"},
    )


def build_online_deployment(model_name: str, model_version: str, settings: AzureMLSettings | None = None) -> ManagedOnlineDeployment:
    settings = settings or get_settings()
    return ManagedOnlineDeployment(
        name=settings.deployment_name,
        endpoint_name=settings.online_endpoint_name,
        model=Model(name=model_name, version=model_version),
        environment=Environment(
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            conda_file={
                "name": "course-endpoint-env",
                "channels": ["conda-forge"],
                "dependencies": [
                    "python=3.10",
                    "pip",
                    {"pip": ["scikit-learn==1.5.2", "joblib==1.4.2", "azureml-inference-server-http"]},
                ],
            },
        ),
        code_configuration=CodeConfiguration(code="src/jobs", scoring_script="score.py"),
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )