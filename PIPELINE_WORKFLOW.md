# MLOps Pipeline Workflow Guide

## Overview

This document provides a comprehensive guide for teammates working on the Heart Disease Prediction MLOps pipeline. It covers the complete workflow from development to production deployment.

## Initial Setup (One-time per teammate)

### Environment Setup

```bash
# Create and activate conda environment
conda create -n myenv python=3.9
conda activate myenv

# Install dependencies
pip install -r requirements.txt

# Verify setup
python run_tests.py  # Should pass all tests
```

**What this does**: Creates isolated environment, installs dependencies, validates setup.

## Development Workflow Commands

### Task 1: Data Acquisition & EDA

```bash
# When dataset changes or EDA needs updates
conda activate myenv
python src/data_acquisition_eda.py

# Test the changes
python run_tests.py --task1
```

**Purpose**:

- Downloads/processes new data
- Updates `data/processed/` files
- Performs exploratory data analysis
- Creates data quality reports

### Task 2: Feature Engineering & Model Development

```bash
# When adding new features or changing algorithms
conda activate myenv
python src/feature_engineering.py

# Test the changes
python run_tests.py --task2
```

**Purpose**:

- Creates new engineered features
- Trains models with new features/algorithms
- Saves models to `models/` directory
- Performs local model evaluation

### Task 3: Experiment Tracking (Production Upload)

```bash
# Upload models to Railway MLflow (creates new versions)
conda activate myenv
python src/experiment_tracking.py

# Test the tracking
python run_tests.py --task3
```

**Purpose**:

- Creates NEW model versions in Railway MLflow
- Uploads models to Railway server
- Registers models with version numbers
- Sets champion model alias
- Enables team collaboration through shared MLflow server

### Task 4: Model Packaging & Deployment Preparation

```bash
# Package the current champion model for deployment
conda activate myenv
python src/model_packaging.py

# Test the packaging
python run_tests.py --task4
```

**Purpose**:

- Downloads champion model from Railway
- Creates deployment packages in multiple formats
- Generates environment snapshots
- Creates configuration files for deployment

## Model Versioning Strategy

### When Model Versions Change

**Model Version INCREASES when:**

```bash
python src/experiment_tracking.py  # Only this command creates new versions
```

**Model Version STAYS SAME when:**

```bash
python src/data_acquisition_eda.py     # Just data preparation
python src/feature_engineering.py      # Local model training only
python src/model_packaging.py          # Uses existing models
python run_tests.py                    # Just testing
```

### Version Progression Example

```
Initial State: No models in Railway
├── Run experiment_tracking.py → Creates Version 1
├── Change features, run experiment_tracking.py → Creates Version 2
├── Run model_packaging.py → Uses Version 2 (no new version)
├── Run tests → Uses Version 2 (no new version)
└── Improve algorithm, run experiment_tracking.py → Creates Version 3
```

## Production Team Workflows

### Scenario 1: Data Scientist Adds New Features

```bash
# 1. Develop new features locally
conda activate myenv
python src/feature_engineering.py  # Test locally first

# 2. Run local tests
python run_tests.py --task2

# 3. Upload to production (creates new model version)
python src/experiment_tracking.py  # Version N → Version N+1

# 4. Package for deployment
python src/model_packaging.py  # Uses Version N+1

# 5. Validate everything
python run_tests.py  # Full pipeline test
```

### Scenario 2: Teammate Wants Latest Models

```bash
# 1. Get latest code
git pull origin main

# 2. Activate environment
conda activate myenv

# 3. Test current setup (uses existing Railway models)
python run_tests.py

# 4. If they want to create NEW models with their changes:
python src/experiment_tracking.py  # Creates new version

# 5. Package their new models
python src/model_packaging.py
```

### Scenario 3: Production Deployment

```bash
# 1. Identify champion model
python src/experiment_tracking.py  # Ensures latest models

# 2. Create deployment package
python src/model_packaging.py  # Creates production package

# 3. Validate package
python run_tests.py --task4

# 4. Deploy (future tasks)
# docker build -t heart-disease-model .
# kubectl apply -f deployment.yaml
```

## Daily Development Workflow

### Morning Routine

```bash
# Get latest changes
git pull
conda activate myenv

# Validate current state
python run_tests.py  # Should pass all tests
```

### Development Cycle

```bash
# Make changes to code
# Edit src/feature_engineering.py or src/data_acquisition_eda.py

# Test locally first
python run_tests.py --task2  # Test your specific changes

# Upload to production when ready
python src/experiment_tracking.py  # Creates new model version

# Package for deployment
python src/model_packaging.py

# Final validation
python run_tests.py  # Full pipeline test

# Commit changes
git add .
git commit -m "Add new features and update models"
git push origin main
```

## Command Reference

| Command                              | Purpose                       | When to Use                     |
| ------------------------------------ | ----------------------------- | ------------------------------- |
| `python src/data_acquisition_eda.py` | Data preparation and analysis | Data changes or new datasets    |
| `python src/feature_engineering.py`  | Local model development       | Feature development and testing |
| `python src/experiment_tracking.py`  | Production model upload       | Ready for production deployment |
| `python src/model_packaging.py`      | Deployment preparation        | Package models for deployment   |
| `python run_tests.py`                | Full pipeline validation      | Testing and CI/CD               |
| `python run_tests.py --task1`        | Test data acquisition         | After data changes              |
| `python run_tests.py --task2`        | Test feature engineering      | After model changes             |
| `python run_tests.py --task3`        | Test experiment tracking      | After MLflow changes            |
| `python run_tests.py --task4`        | Test model packaging          | After packaging changes         |

## Railway MLflow Integration

### Access

- **URL**: https://mlflow-tracking-production-53fb.up.railway.app
- **Purpose**: Shared team experiment tracking and model registry

### Team Capabilities

- View all experiments and model versions
- Compare model performance across versions
- Download models for local testing
- Set champion/staging/production aliases
- Track model lineage and metadata
- Collaborate on model development

### Model Registry Structure

```
Railway MLflow Server:
├── heart_disease_random_forest: Version N
├── heart_disease_logistic_regression: Version N
├── heart_disease_gradient_boosting: Version N
└── heart_disease_svm: Version N
```

## Production Monitoring

### Check Current Production Model

```bash
python -c "
from src.model_packaging import ModelPackager
packager = ModelPackager()
info = packager.get_champion_model_info()
print(f'Production Model: {info[\"model_name\"]} v{info[\"version\"]}')
print(f'Performance: {info[\"roc_auc\"]:.4f}')
"
```

### Validate Production Package

```bash
python run_tests.py --task4
```

### Create New Deployment Package

```bash
python src/model_packaging.py
```

## API Feature Engineering (TASK-6 Fix)

### Problem Solved

The CI/CD pipeline was failing with error: "The feature names should match those that were passed during fit. Feature names seen at fit time, yet now missing: age_group, age_sex_interaction, chol_age_ratio..."

### Root Cause

The model was trained with engineered features (6 additional features created during Task 2), but the API was only receiving the raw 13 input features. This caused a feature mismatch error during prediction.

### Solution Implemented

The API now automatically applies the same feature engineering pipeline used during model training:

**File**: `app/prediction.py`

**Engineered Features Applied**:

1. **age_group**: Categorizes age into young (0-40), middle_aged (40-50), senior (50-60), elderly (60+)
2. **chol_age_ratio**: Metabolic health indicator = cholesterol / age
3. **heart_rate_reserve**: Exercise capacity = max_heart_rate - (220 - age)
4. **risk_score**: Composite risk = age*0.1 + chol*0.01 + trestbps*0.1 + oldpeak*10
5. **age_sex_interaction**: Combined effect = age \* sex
6. **cp_exang_interaction**: Combined effect = chest_pain_type \* exercise_angina

**How It Works**:

```
API Input (13 raw features)
    ↓
_engineer_features() method
    ↓
Creates 6 engineered features
    ↓
Combined 19 features (13 raw + 6 engineered)
    ↓
Model prediction
    ↓
API Output
```

**API Endpoints Updated**:

- `POST /predict`: Automatically applies feature engineering before prediction
- `GET /model/info`: Returns both raw_feature_names and engineered_feature_names

**Testing**:

The fix ensures CI/CD pipeline tests pass by providing models with the exact feature space they were trained on.

## File Structure Overview

```
Assignment-1/
├── src/                          # Source code
│   ├── data_acquisition_eda.py   # Task 1: Data loading and EDA
│   ├── feature_engineering.py    # Task 2: Feature engineering and modeling
│   ├── experiment_tracking.py    # Task 3: MLflow experiment tracking
│   └── model_packaging.py        # Task 4: Model packaging and deployment
├── data/                         # Data storage
│   ├── processed/               # Processed datasets
│   └── raw/                     # Raw data files
├── models/                       # Local model artifacts
├── packages/                     # Deployment packages
├── scripts/                      # Testing and utility scripts
├── tests/                        # Unit tests
├── environments/                 # Environment specifications
├── requirements.txt             # Python dependencies
└── run_tests.py                 # Test runner
```

## Best Practices

### Code Changes

1. Always test locally first with individual task tests
2. Run full test suite before committing
3. Use descriptive commit messages
4. Create new model versions only when ready for production

### Model Development

1. Develop features in Task 2 first
2. Test thoroughly with local validation
3. Upload to Railway only when confident
4. Package models after validation

### Team Collaboration

1. Pull latest changes before starting work
2. Communicate model version changes to team
3. Use Railway MLflow UI for model comparison
4. Document significant changes in commit messages

### Production Deployment

1. Always validate packages before deployment
2. Monitor model performance in production
3. Keep deployment packages versioned
4. Maintain environment reproducibility

## Troubleshooting

### Common Issues

- **Import errors**: Ensure `conda activate myenv` is run
- **MLflow connection**: Check Railway server status
- **Model loading**: Verify model versions exist in Railway
- **Package validation**: Check all required files are present

### Debug Commands

```bash
# Check MLflow connection
python -c "import mlflow; mlflow.set_tracking_uri('https://mlflow-tracking-production-53fb.up.railway.app'); print('Connected')"

# List available models
python -c "from mlflow import MlflowClient; client = MlflowClient(); print([m.name for m in client.search_registered_models()])"

# Check package contents
ls -la packages/heart_disease_model_*/
```

This workflow ensures controlled model versioning, team collaboration, and production readiness while maintaining clear separation between development, testing, and production deployment phases.

## CI/CD Workflow Dependencies (TASK-6 Optimization)

### Problem Solved

Two separate workflows were running simultaneously and failing at the same time:

- **CI Pipeline - Lint and Test** (`ci.yml`) - had container testing
- **Build and Push Container** (`container-build.yml`) - also had container testing

This caused resource conflicts and duplicate testing efforts.

### Solution Implemented

Restructured workflows with proper sequential dependencies:

**Workflow Execution Order**:

```
1. CI Pipeline (ci.yml) - Runs on push
   ├── lint (Code Quality Checks)
   ├── test (Unit Tests) - depends on lint
   └── Completes

2. Build and Push Container (container-build.yml) - Triggered after CI completes
   └── build-and-test (Build and Test Container) - depends on CI success
       └── Completes
```

**Changes Made**:

1. **ci.yml**:

   - Removed redundant `container-test` job
   - Focuses on code quality and unit testing only
   - Runs on every push to main/develop

2. **container-build.yml**:
   - Added `workflow_run` trigger to depend on CI Pipeline completion
   - Added condition: `if: github.event.workflow_run.conclusion == 'success' || github.event_name != 'workflow_run'`
   - Only runs container build/test after CI passes
   - Prevents duplicate container testing

**Benefits**:

- No more simultaneous job failures
- Clear sequential workflow execution
- Faster feedback (CI runs first, container build only if CI passes)
- Reduced resource usage (no duplicate container tests)
- Better error isolation (know exactly which stage failed)

**Workflow Triggers**:

- **CI Pipeline**: Push to main/develop, Pull Requests
- **Container Build**: After CI completes successfully, Manual dispatch, Tags
