# MLOps CI Pipeline Documentation

## Overview
This project uses GitHub Actions for Continuous Integration.
The workflow runs automatically on push and pull requests to main and develop.

## Workflow Jobs

### 1. build-and-lint
- Checks out the repository.
- Sets up Python 3.11.
- Installs project dependencies.
- Compiles Python files.
- Runs Flake8 lint checks.

### 2. test-suite
- Runs the data collection, preprocessing and feature engineering pipeline.
- Executes pytest test cases.
- Generates coverage.xml and test-results.xml.
- Uploads test reports as GitHub Actions artifacts.

### 3. model-quality-gate
- Runs after test-suite succeeds.
- Executes model quality tests.
- Verifies the model meets the required quality criteria.

## Job Dependencies
build-and-lint -> test-suite -> model-quality-gate

## Trigger
The workflow runs on push and pull_request events for main and develop branches.

## Verification
The GitHub Actions run completed successfully.
All three jobs passed:
- build-and-lint
- test-suite
- model-quality-gate

## Conclusion
GitHub Actions automates code compilation, linting, testing and model quality validation.
