import mlflow
from mlflow.tracking import MlflowClient


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "iris-classification-baseline"
MODEL_NAME = "iris-classifier-prod"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()


# Get experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")


# Find the run with the highest F1 score
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_macro DESC"],
    max_results=1,
)

if not runs:
    raise ValueError("No MLflow runs found.")


best_run = runs[0]

run_id = best_run.info.run_id
best_f1 = best_run.data.metrics["f1_macro"]

print("Best Run ID:", run_id)
print("Best F1 Score:", best_f1)


# Register model
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)

print("Registered Model:", MODEL_NAME)
print("Model Version:", result.version)


# Move model to Staging
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Staging",
)

print(f"Model version {result.version} moved to Staging.")