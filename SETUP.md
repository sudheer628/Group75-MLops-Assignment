# Setup Guide

In this guide, We explain how to set up the development environment for our Heart Disease Prediction project. We have tried to make it as simple as possible - should take about 5 minutes.

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/sudheer628/group75-mlops-assignment.git
cd group75-mlops-assignment
```

### 2. Create Conda Environment

We used Python 3.12 for this project. Create the conda environment with:

```bash
conda create -n myenv python=3.12
conda activate myenv
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Everything Works

Run the test suite to make sure everything is set up correctly:

```bash
python run_tests.py
```

If everything is working, you should see output like this:

```
STEP ENV: Environment Validation
Environment      PASS
Dependencies     PASS
MLflow           PASS
Structure        PASS

Environment validation successful!

Running Task 1 Tests...
All Task 1 tests passed!

Running Task 2 Tests...
All Task 2 tests passed!

Running Task 3 Tests...
All Task 3 tests passed!

Running Task 4 Tests...
All Task 4 tests passed!

ALL TESTS PASSED!
```

---

## Running Tests

We have unit tests for each task in the `tests/` folder. Here's how to run them:

```bash
# Run all tests
python run_tests.py

# Run tests for a specific task
python run_tests.py --task1    # Data acquisition tests
python run_tests.py --task2    # Feature engineering tests
python run_tests.py --task3    # MLflow tracking tests
python run_tests.py --task4    # Model packaging tests

# Just validate environment (quick check)
python run_tests.py --validate-only
```

The tests also run automatically in our CI/CD pipeline whenever we push code to GitHub.

---

## Running Individual Tasks

Each task has its own Python script in the `src/` folder:

```bash
# Make sure environment is activated first
conda activate myenv

# Task 1: Data Acquisition & EDA
python src/data_acquisition_eda.py

# Task 2: Feature Engineering & Model Training
python src/feature_engineering.py

# Task 3: Upload to MLflow (Railway)
python src/experiment_tracking.py

# Task 4: Package Models for Deployment
python src/model_packaging.py

# Validate Feature Store (runs in CI/CD too)
python src/feature_store.py --validate
```

---

## Common Issues

### "conda environment not activated"

Make sure to run `conda activate myenv` before any Python commands.

### "ModuleNotFoundError"

Install dependencies again:

```bash
pip install -r requirements.txt
```

### "MLflow connection failed"

Check your internet connection. The MLflow server is hosted on Railway at:
https://mlflow-tracking-production-53fb.up.railway.app

---

## Project URLs

Once set up, you can access:

- **Live API**: http://myprojectdemo.online
- **API Docs**: http://myprojectdemo.online/docs
- **MLflow Dashboard**: https://mlflow-tracking-production-53fb.up.railway.app
- **Grafana Monitoring**: https://group75mlops.grafana.net

---

## What Gets Created

After running the tasks, these directories will have generated files:

- `data/processed/` - Cleaned and processed datasets
- `models/` - Trained model files (.joblib)
- `feature_store/` - Feature schemas and metadata
- `packages/` - Deployment packages (gitignored)
- `environments/` - Environment snapshots (gitignored)

The processed data and models are committed to the repo, but packages and environments are regenerated as needed.
