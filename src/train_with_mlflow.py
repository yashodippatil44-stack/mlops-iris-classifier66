import os

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# MLflow configuration
# -----------------------------
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(TRACKING_URI)

EXPERIMENT_NAME = "iris-classification-baseline"
mlflow.set_experiment(EXPERIMENT_NAME)


# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "data/processed/iris_features.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# -----------------------------
# Features and target
# -----------------------------
feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]

X = df[feature_cols]
y = df["species"]


# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)


# -----------------------------
# Models
# -----------------------------
models = [
    (
        "logistic_regression",
        LogisticRegression(
            max_iter=200,
            C=1.0
        ),
        {
            "max_iter": 200,
            "C": 1.0
        },
    ),
    (
        "random_forest_shallow",
        RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        ),
        {
            "n_estimators": 50,
            "max_depth": 3
        },
    ),
    (
        "random_forest_deep",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        ),
        {
            "n_estimators": 200,
            "max_depth": None
        },
    ),
]


# -----------------------------
# Train and track models
# -----------------------------
results = []

for model_name, model, params in models:

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            average="macro"
        )
        recall = recall_score(
            y_test,
            y_pred,
            average="macro"
        )
        f1 = f1_score(
            y_test,
            y_pred,
            average="macro"
        )

        # Log parameters
        mlflow.log_param("model_type", model_name)

        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1)

        # -----------------------------
        # Confusion matrix artifact
        # -----------------------------
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=label_encoder.classes_
        )

        disp.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        cm_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        # -----------------------------
        # Log model
        # -----------------------------
        mlflow.sklearn.log_model(
            model,
            "model"
        )

        # Save result
        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
            }
        )

        print(f"\nModel: {model_name}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")


# -----------------------------
# Compare models
# -----------------------------
results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df)

best_model = results_df.loc[
    results_df["f1_macro"].idxmax()
]

print("\n===== BEST MODEL =====")
print(best_model)