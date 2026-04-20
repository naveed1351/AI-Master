from __future__ import annotations

from pathlib import Path

from azure.ai.ml import Input, automl, command, sweep
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.sweep import Choice, Uniform

from .config import AzureMLSettings, get_settings


def build_command_training_job(settings: AzureMLSettings | None = None):
    settings = settings or get_settings()
    return command(
        code=str(Path(__file__).resolve().parents[1]),
        command="python jobs/train_iris.py --learning-rate ${{inputs.learning_rate}} --n-estimators ${{inputs.n_estimators}}",
        environment="AzureML-sklearn-1.5:1",
        compute=settings.compute_name,
        experiment_name="azureml-course-command-job",
        display_name="train-iris-model",
        inputs={
            "learning_rate": 0.05,
            "n_estimators": 200,
        },
    )


def build_sweep_job(settings: AzureMLSettings | None = None):
    base_job = build_command_training_job(settings)
    return sweep(
        base_job,
        sampling_algorithm="random",
        primary_metric="accuracy",
        goal="maximize",
        max_total_trials=6,
        max_concurrent_trials=2,
        trial_timeout=1800,
        search_space={
            "learning_rate": Uniform(min_value=0.01, max_value=0.2),
            "n_estimators": Choice(values=[50, 100, 200, 300]),
        },
    )


def build_automl_classification_job(data_path: str, settings: AzureMLSettings | None = None):
    settings = settings or get_settings()
    return automl.classification(
        compute=settings.compute_name,
        experiment_name="azureml-course-automl",
        training_data=Input(path=data_path, type=AssetTypes.URI_FILE),
        target_column_name="species",
        primary_metric="accuracy",
        n_cross_validations=3,
        enable_model_explainability=True,
        tags={"course": "azure-machine-learning", "module": "automl"},
    )