
"""
Validation stage for the Iris ML pipeline.
Checks required columns, species labels, and numeric ranges.
"""

import logging

import pandas as pd


logger = logging.getLogger("validate")


class DataValidationError(Exception):
    """Raised when input data fails validation."""


REQUIRED_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "species",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
    "petal_length_bin",
]

VALID_SPECIES = {"setosa", "versicolor", "virginica"}

NUMERIC_RANGES = {
    "sepal length (cm)": (3.0, 9.0),
    "sepal width (cm)": (1.0, 6.0),
    "petal length (cm)": (0.0, 8.0),
    "petal width (cm)": (0.0, 3.0),
}


def validate(input_path):
    """Validate a processed feature CSV and return its DataFrame."""
    df = pd.read_csv(input_path)
    errors = []

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        errors.append(f"Missing expected columns: {missing_columns}")

    if "species" in df.columns:
        unexpected_species = set(df["species"].dropna().unique()) - VALID_SPECIES

        if unexpected_species:
            errors.append(f"Unexpected species values: {unexpected_species}")

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        invalid_mask = (
            values.notna()
            & ((values < minimum) | (values > maximum))
        )

        if invalid_mask.any():
            errors.append(
                f"{int(invalid_mask.sum())} rows out of expected range "
                f"for '{column}' ({minimum}-{maximum})"
            )

    if errors:
        logger.error("; ".join(errors))
        raise DataValidationError(
            f"Validation failed with {len(errors)} error(s): "
            + "; ".join(errors)
        )

    logger.info("Validation passed for %s rows", len(df))
    return df
