
import mlflow
import pandas as pd

EXPERIMENT_NAME = "iris-hyperparameter-tuning"


def compare_results():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        print("Experiment not found in MLflow.")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas",
    )

    if runs.empty:
        print("No runs found in this experiment.")
        return

    rows = []

    for _, run in runs.iterrows():
        run_name = run.get("tags.mlflow.runName", "Unknown")
        metrics = run.get("metrics.test_accuracy")
        cv_score = run.get("metrics.best_cv_f1_macro")

        # Baseline script may log its CV score under a different metric name.
        if pd.isna(cv_score):
            cv_score = run.get("metrics.cv_f1_macro_mean")

        rows.append({
            "Run Name": run_name,
            "CV F1 Macro": cv_score,
            "Test Accuracy": metrics,
            "Run ID": run.get("run_id"),
        })

    results = pd.DataFrame(rows)
    results = results.sort_values(
        by="Test Accuracy", ascending=False, na_position="last"
    )

    print("\n=== Hyperparameter Tuning Comparison ===")
    print(results.to_string(index=False))

    results.to_csv("tuning_comparison_results.csv", index=False)
    print("\nComparison saved to tuning_comparison_results.csv")


if __name__ == "__main__":
    compare_results()
