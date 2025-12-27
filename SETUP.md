# New Teammate Setup Guide

## Quick Start (5 minutes)

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Create conda environment
conda create -n myenv python=3.9
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

### 2. Validate Setup

```bash
# Run comprehensive validation + all tests (recommended)
python run_tests.py

# OR run only validation (quick environment check)
python run_tests.py --validate-only
```

**Expected Result**: All validation steps should show ✅ PASS

### 3. What You Should See

```
STEP ENV: Environment Validation
Environment      ✅ PASS
Dependencies     ✅ PASS
MLflow          ✅ PASS
Structure       ✅ PASS

🎉 Environment validation successful!

Running Task 1 Tests...
All Task 1 tests passed!

Running Task 2 Tests...
All Task 2 tests passed!

Running Task 3 Tests...
All Task 3 tests passed!

Running Task 4 Tests...
All Task 4 tests passed!

🎉 ALL TESTS PASSED! Your implementation is working correctly.
```

## If Setup Fails

### Common Issues & Solutions

**❌ Environment not activated**

```bash
conda activate myenv
```

**❌ Missing dependencies**

```bash
pip install -r requirements.txt
```

**❌ Environment validation failed**

```bash
# Run validation only to see specific issues
python run_tests.py --validate-only
```

**❌ Import errors**

- Make sure `conda activate myenv` is run before any Python commands
- Verify all packages installed: `pip list`

## What Gets Created

After successful setup, you'll have:

- ✅ `data/processed/` - Processed datasets
- ✅ `models/` - Trained models
- ✅ Railway MLflow connection - Shared experiment tracking
- ✅ Complete test suite passing

## Next Steps

1. **Read the workflow guide**: `PIPELINE_WORKFLOW.md`
2. **Access MLflow UI**: https://mlflow-tracking-production-53fb.up.railway.app
3. **Start developing**: `python src/experiment_tracking.py`

## Team Commands Reference

```bash
# Always activate environment first
conda activate myenv

# Run validation + all tests (comprehensive)
python run_tests.py

# Run only environment validation (quick check)
python run_tests.py --validate-only

# Skip validation, run tests only (not recommended)
python run_tests.py --skip-validation

# Individual task tests
python run_tests.py --task1             # Test Task 1
python run_tests.py --task2             # Test Task 2
python run_tests.py --task3             # Test Task 3
python run_tests.py --task4             # Test Task 4

# Individual tasks (development)
python src/data_acquisition_eda.py      # Task 1: Data & EDA
python src/feature_engineering.py       # Task 2: Models (local)
python src/experiment_tracking.py       # Task 3: Upload to Railway
python src/model_packaging.py           # Task 4: Package for deployment
```

## Important Notes

- **Always use `conda activate myenv`** before running any commands
- **Model versions only change** when running `python src/experiment_tracking.py`
- **Railway MLflow** is shared - all teammates see the same experiments
- **Generated files** (packages/, environments/) are gitignored - they regenerate automatically

## Help & Documentation

- **Complete workflow**: `PIPELINE_WORKFLOW.md`
- **Project overview**: `README.md`
- **Assignment details**: `INSTRUCTIONS.md`
- **Implementation plan**: `PLAN.md`

---

**Setup Time**: ~5 minutes  
**First Test Run**: ~3 minutes (downloads data, trains models)  
**Subsequent Runs**: ~30 seconds
