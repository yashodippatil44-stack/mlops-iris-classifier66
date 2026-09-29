import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path=".")

entity_df = pd.read_parquet("data/iris_features.parquet")[
    ["sample_id", "event_timestamp"]
].head(5)

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_measurements:sepal_length_cm",
        "iris_measurements:sepal_width_cm",
        "iris_measurements:petal_length_cm",
        "iris_measurements:petal_width_cm",
        "iris_engineered_features:sepal_area",
        "iris_engineered_features:petal_area",
        "iris_engineered_features:sepal_to_petal_length_ratio",
        "iris_engineered_features:petal_length_bin",
    ],
).to_df()

print(training_df)
print("\nRows:", len(training_df))