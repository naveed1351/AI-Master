# Azure Machine Learning Course

This course turns the repository into a structured Azure Machine Learning learning path for beginners, developers, and ML engineers who want hands-on examples with the Python SDK v2.

## What You Will Learn

- What Azure Machine Learning is and where it fits in the ML lifecycle
- How to connect to a workspace with the Python SDK v2
- How to create data assets, environments, and compute targets
- How to submit training jobs and track experiments
- How to use AutoML and MLflow together
- How to build reusable pipelines and components
- How to deploy models to managed online endpoints
- How to think about MLOps, security, and responsible AI in Azure ML

## Course Structure

1. `notebooks/01_introduction_to_azure_ml.ipynb`
2. `notebooks/02_workspace_and_compute.ipynb`
3. `notebooks/03_data_and_training_jobs.ipynb`
4. `notebooks/04_automl_and_mlflow.ipynb`
5. `notebooks/05_pipelines_and_components.ipynb`
6. `notebooks/06_model_deployment.ipynb`
7. `notebooks/07_mlop_security_and_responsible_ai.ipynb`
8. `notebooks/08_capstone_project.ipynb`

## Python Examples

The `src/` folder contains reusable modules and runnable examples:

- `src/azureml_course/config.py`: environment-based configuration
- `src/azureml_course/workspace.py`: workspace connection helpers
- `src/azureml_course/dataset_utils.py`: sample data creation helpers
- `src/azureml_course/training_jobs.py`: job builders for command, sweep, and AutoML
- `src/azureml_course/pipelines.py`: reusable training pipeline example
- `src/azureml_course/deployment_samples.py`: online endpoint examples
- `src/jobs/`: runnable training, prep, evaluation, and scoring scripts

## Suggested Learning Flow

1. Start with notebook 1 to understand the service and core concepts.
2. Move to notebooks 2 and 3 to connect a workspace and run jobs.
3. Use notebooks 4 through 6 to learn AutoML, MLflow, pipelines, and deployment.
4. Finish with notebooks 7 and 8 for production concerns and a capstone exercise.

## Prerequisites

- An Azure subscription
- Permission to create or use an Azure Machine Learning workspace
- Python 3.10+
- Azure CLI logged in with `az login`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Populate `.env` with your subscription, resource group, workspace, and optional endpoint names.

## References Used

- Azure ML examples: <https://github.com/Azure/azureml-examples>
- Product overview: <https://azure.microsoft.com/en-us/products/machine-learning>
- What is Azure Machine Learning: <https://learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learning?view=azureml-api-2>
- Azure ML documentation hub: <https://learn.microsoft.com/en-us/azure/machine-learning/?view=azureml-api-2>
