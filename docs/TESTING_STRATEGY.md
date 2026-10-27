\# Testing Strategy â€” Experiment 8



\## 1. Objective



The objective is to test data preprocessing, feature engineering, data validation, training pipeline behavior, and model output quality using automated pytest tests.



\## 2. Test Coverage



| Test File | Layer Tested | Number of Tests |

|---|---|---:|

| `test\_preprocess.py` | Data preprocessing | 4 |

| `test\_features.py` | Feature engineering | 3 |

| `test\_validate.py` | Data validation and negative cases | 4 |

| `test\_training\_pipeline.py` | Model training and prediction smoke tests | 3 |

| `test\_model\_quality.py` | Prediction validity and model accuracy | 2 |

| \*\*Total\*\* | | \*\*16\*\* |



\## 3. Coverage Results



Coverage was measured using:



`pytest --cov=src --cov-report=term-missing --junitxml=test-results.xml`



\- Total source-code coverage: \*\*18%\*\*

\- `src/pipeline/preprocess.py`: \*\*83%\*\*

\- `src/pipeline/features.py`: \*\*75%\*\*

\- `src/pipeline/validate.py`: \*\*97%\*\*

\- Tests passed: \*\*16 out of 16\*\*

\- Test failures: \*\*0\*\*

\- Execution time: \*\*6.40 seconds\*\*



The overall coverage is limited because several training, tuning, data collection, and model registration scripts were not executed by these unit tests.



\## 4. Testing Approach



\- \*\*Unit testing:\*\* Checks individual preprocessing, feature engineering, and validation functions.

\- \*\*Smoke testing:\*\* Checks that a model can train and produce predictions with the expected shape.

\- \*\*Model output testing:\*\* Checks prediction labels and verifies that predicted probabilities sum to approximately 1.

\- \*\*Model quality testing:\*\* Checks whether test accuracy meets the minimum threshold of 0.90.

\- \*\*Negative testing:\*\* Verifies that validation rejects missing columns, unexpected species labels, and out-of-range values.



\## 5. Test Reports



The JUnit XML report is generated as `test-results.xml`. The pytest coverage report shows statement counts, missed lines, and coverage percentages.



\## 6. Conclusion



All 16 tests passed successfully. The tests cover important data-processing, validation, training-smoke-test, and model-quality behaviors. The measured overall source-code coverage is 18%, with high coverage in the tested validation and preprocessing modules.
