from __future__ import annotations

from pathlib import Path

from azure.ai.ml import Input, command, dsl
from azure.ai.ml.constants import AssetTypes

from .config import AzureMLSettings, get_settings


def build_training_pipeline(settings: AzureMLSettings | None = None):
    settings = settings or get_settings()
    src_dir = str(Path(__file__).resolve().parents[1])

    prep_component = command(
        code=src_dir,
        command="python jobs/prep_data.py --output-dir ${{outputs.output_dir}}",
        environment="AzureML-sklearn-1.5:1",
        outputs={"output_dir": {"mode": "rw_mount"}},
        name="prep_iris_data_component",
    )

    train_component = command(
        code=src_dir,
        command="python jobs/train_iris.py --train-data ${{inputs.train_data}}/train.csv --model-output ${{outputs.model_output}}",
        environment="AzureML-sklearn-1.5:1",
        inputs={"train_data": Input(type=AssetTypes.URI_FOLDER)},
        outputs={"model_output": {"mode": "rw_mount"}},
        name="train_iris_component",
    )

    @dsl.pipeline(compute=settings.compute_name, description="Azure ML course training pipeline")
    def iris_pipeline():
        prep_node = prep_component()
        train_node = train_component(train_data=prep_node.outputs.output_dir)
        return {"model_output": train_node.outputs.model_output}

    pipeline_job = iris_pipeline()
    pipeline_job.experiment_name = "azureml-course-pipeline"
    return pipeline_job