
# Hyperparameter Tuning Analysis

## 1. Objective

The objective of this experiment is to develop a baseline machine learning model, apply Grid Search and Random Search for hyperparameter tuning, and compare their performance using cross-validation and test accuracy.

## 2. Dataset and Evaluation

- Dataset: Iris dataset with engineered features.
- Target column: `species`
- Train-test split: 80% training and 20% testing.
- Random state: 42
- Cross-validation: 5-fold
- Evaluation metric: Macro F1-score
- Experiment tracking: MLflow
- MLflow experiment name: `iris-hyperparameter-tuning`

## 3. Baseline Model

A Decision Tree Classifier was used as the baseline model.

Results:
- Cross-validation Macro F1-score: 0.9663
- Test accuracy: 90.00%

The baseline provides a reference point for evaluating the tuned models.

## 4. Grid Search

Grid Search was performed using a Random Forest Classifier.

Hyperparameters explored:
- `n_estimators`: 50, 100, 200
- `max_depth`: 3, 5, 10, None
- `min_samples_split`: 2, 5, 10
- `max_features`: sqrt, log2

Results:
- Parameter combinations: 72
- Cross-validation folds: 5
- Total model fits: 360
- Best parameters:
  - `max_depth`: 3
  - `max_features`: sqrt
  - `min_samples_split`: 2
  - `n_estimators`: 50
- Best CV Macro F1-score: 0.9663
- Test accuracy: 96.67%

## 5. Random Search

Random Search was performed using a Random Forest Classifier.

Search configuration:
- `n_estimators`: Random integer from 50 to 299
- `max_depth`: 3, 5, 10, 15, None
- `min_samples_split`: Random integer from 2 to 14
- `max_features`: sqrt, log2
- Number of sampled configurations: 30
- Cross-validation folds: 5
- Total model fits: 150

Best parameters:
- `max_depth`: 3
- `max_features`: sqrt
- `min_samples_split`: 6
- `n_estimators`: 100

Results:
- Best CV Macro F1-score: 0.9663
- Test accuracy: 96.67%

## 6. Comparative Performance Analysis

| Model | CV Macro F1-score | Test Accuracy | Model Fits |
|---|---:|---:|---:|
| Baseline Decision Tree | 0.9663 | 90.00% | Not applicable |
| Grid Search Random Forest | 0.9663 | 96.67% | 360 |
| Random Search Random Forest | 0.9663 | 96.67% | 150 |

### Observations

1. Both Grid Search and Random Search achieved the same test accuracy of 96.67%.
2. Both tuning approaches achieved a CV Macro F1-score of approximately 0.9663.
3. Grid Search evaluated 72 parameter combinations using 5-fold cross-validation, resulting in 360 model fits.
4. Random Search evaluated 30 sampled configurations using 5-fold cross-validation, resulting in 150 model fits.
5. Random Search used fewer model fits while achieving the same observed test accuracy as Grid Search.
6. Both tuned Random Forest models achieved higher test accuracy than the baseline Decision Tree in this experiment.

## 7. MLflow Tracking

The experiment was tracked using MLflow under:

`iris-hyperparameter-tuning`

Three runs were recorded:
- `baseline_decision_tree`
- `grid_search_random_forest`
- `random_search_random_forest`

The comparison results were exported to:

`tuning_comparison_results.csv`

The candidate-level results were exported to:
- `grid_search_all_candidates.csv`
- `random_search_all_candidates.csv`

## 8. Conclusion

In this experiment, both Grid Search and Random Search improved test accuracy from 90.00% for the baseline Decision Tree to 96.67% for the tuned Random Forest models.

Random Search achieved the same observed test accuracy as Grid Search with 150 model fits instead of 360. Therefore, Random Search was more computationally economical for this particular experiment.

These results apply to the current dataset split and configuration; they do not guarantee that Random Search will always outperform Grid Search.
