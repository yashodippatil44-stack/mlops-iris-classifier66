import mlflow
import mlflow.sklearn
import pandas as pd


# MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Load the registered model from Staging
model_uri = "models:/iris-classifier-prod/Staging"

model = mlflow.sklearn.load_model(model_uri)

print("Model loaded successfully!")


# Load test data
df = pd.read_csv("data/processed/iris_features.csv")


# Features used during training
feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]


# Take one sample
sample = df[feature_cols].iloc[[0]]

# Predict
prediction = model.predict(sample)

print("Input sample:")
print(sample)

print("\nPredicted class:", prediction[0])