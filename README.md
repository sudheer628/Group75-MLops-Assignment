# Heart Disease Prediction MLOps Project

This project implements an end-to-end MLOps pipeline for heart disease prediction using the UCI Heart Disease dataset.

## Project Structure

```
├── src/                          # Source code
│   ├── data_acquisition_eda.py   # Task 1: Data acquisition and EDA
│   ├── feature_engineering.py    # Task 2: Feature engineering and model development
│   ├── experiment_tracking.py    # Task 3: MLflow experiment tracking
│   └── model_packaging.py        # Task 4: Model packaging and reproducibility
├── data/                         # Data storage (gitignored except structure)
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed datasets
├── figures/                      # EDA visualizations (gitignored except structure)
├── models/                       # Trained models (gitignored except structure)
├── tests/                        # Unit tests
├── logs/                         # Application logs (gitignored except structure)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Testing Instructions

### Quick Test (Recommended)

Run all tests with the simple test runner:

**IMPORTANT: Always activate conda environment first:**

```bash
conda activate myenv
```

### Quick Test (Recommended)

```bash
conda activate myenv
python run_tests.py
```

### Individual Task Testing

Test only Task 1:

```bash
conda activate myenv
python run_tests.py --task1
```

Test only Task 2:

```bash
conda activate myenv
python run_tests.py --task2
```

Test only Task 3:

```bash
conda activate myenv
python run_tests.py --task3
```

Test only Task 4:

```bash
conda activate myenv
python run_tests.py --task4
```

### Manual Testing

Run individual task files directly:

```bash
conda activate myenv

# Run Task 1
python src/data_acquisition_eda.py

# Run Task 3
python src/experiment_tracking.py

# Test Task 1
python tests/test_task1_data_acquisition.py

# Test Task 2
python tests/test_task2_feature_engineering.py

# Test Task 3
python tests/test_task3_experiment_tracking.py

# Test Task 4
python tests/test_task4_model_packaging.py
```

### What the Tests Check

**Task 1 Tests:**

- Dataset loading from UCI or local files
- Data quality assessment (missing values, duplicates)
- EDA analysis functionality
- Data saving and file creation
- Data consistency and value ranges
- Full pipeline execution

**Task 3 Tests:**

- MLflow Railway server connection and setup
- Experiment creation and management (mocked to avoid server pollution)
- Data information logging and tracking
- Feature engineering information logging
- Model training and logging with Railway compatibility
- Experiment results viewing and analysis
- Railway server integration validation
- Full pipeline execution with remote tracking

**Task 4 Tests:**

- MLflow Railway server connection and registry access
- Champion model identification from Railway registry
- Preprocessing pipeline creation and validation
- Custom transformer functionality (missing value handling, feature engineering)
- Environment snapshot creation (pip, conda, system info)
- Configuration file generation (model config, deployment config)
- Package validation and testing
- Complete deployment package creation
- Full pipeline execution with Railway integration

**Task 2 Tests:**

- Processed data loading
- Missing value handling
- Target variable preparation (binary conversion)
- Feature engineering (6 new features)
- Preprocessing pipeline creation
- Data splitting with stratification
- Model definition and training
- Cross-validation evaluation
- Model evaluation and metrics
- Model persistence and saving
- Full pipeline execution

## Setup Instructions for New Teammates

> **Quick Setup Guide**: See [SETUP.md](SETUP.md) for a streamlined 5-minute setup process.

### Prerequisites

- Python 3.9+
- Conda package manager
- Git

### Fresh Workstation Setup (Step-by-Step)

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create conda environment**

   ```bash
   # Create new environment with Python 3.9
   conda create -n myenv python=3.9

   # Activate the environment
   conda activate myenv
   ```

3. **Install dependencies**

   ```bash
   # Install all required packages
   pip install -r requirements.txt
   ```

4. **Verify setup with comprehensive validation**

   ```bash
   # Run comprehensive validation + all tests (recommended)
   python run_tests.py

   # OR run only validation (quick environment check)
   python run_tests.py --validate-only
   ```

### Expected Test Results (Fresh Setup)

When running `python run_tests.py` on a fresh workstation, you should see:

```
✓ Task 1 Tests: PASS (downloads data, runs EDA)
✓ Task 2 Tests: PASS (trains models locally)
✓ Task 3 Tests: PASS (connects to Railway MLflow)
✓ Task 4 Tests: PASS (packages models from Railway)
```

**Note**: First run will download the UCI dataset (~1MB) and may take 2-3 minutes for model training.

### Troubleshooting Common Issues

**Import Errors**:

```bash
# Make sure conda environment is activated
conda activate myenv
```

**MLflow Connection Issues**:

```bash
# Test Railway connection
python -c "import mlflow; mlflow.set_tracking_uri('https://mlflow-tracking-production-53fb.up.railway.app'); print('Connected')"
```

**Missing Data**:

```bash
# Run data acquisition first
python src/data_acquisition_eda.py
```

**IMPORTANT: Always use the conda environment 'myenv' for all commands to avoid import errors.**

## Model Storage Strategy

This project uses a **comprehensive model storage approach** with Railway MLflow integration:

### **Task 2: Local Model Storage (`models/` directory)**

- **Purpose**: Quick access, backup, and standalone model files
- **Format**: Joblib serialization (fast, scikit-learn optimized)
- **Location**: `models/` directory (local)
- **Files**:
  - `best_model.joblib` - Highest performing model
  - `{algorithm}_model.joblib` - Individual algorithm models
  - `evaluation_results.json` - Performance comparison
  - `feature_names.json` - Feature metadata

### **Task 3: Railway MLflow Server (Primary Storage)**

- **Purpose**: Production experiment tracking, model versioning, and team collaboration
- **Format**: MLflow model format with complete metadata
- **Location**: Railway MLflow server (remote, shared)
- **URL**: `https://mlflow-tracking-production-53fb.up.railway.app`
- **Features**:
  - **Model artifacts storage** (models stored on Railway server)
  - **Model versioning and registry** (with Railway 2.10.0 compatibility)
  - **Complete experiment metadata** (parameters, metrics, artifacts)
  - **Team collaboration** (shared access for all teammates)
  - **Performance metrics history** (cross-validation, model comparison)
  - **Experiment organization** (tags, stages, model families)

### **Railway MLflow Integration Details**

**Compatibility Solution**: MLflow 3.8.0 client ↔ Railway 2.10.0 server

- Uses compatibility approach: `mlflow.sklearn.save_model()` + `mlflow.log_artifacts()`
- Avoids new MLflow 3.x API endpoints that don't exist in Railway 2.10.0
- Successfully stores both model artifacts and experiment metadata

**Team Access**:

- All teammates can access the same MLflow server
- Shared experiments and model registry
- No local database conflicts or synchronization issues
- Professional experiment tracking interface

### **Model Access Patterns**

```python
# Quick model loading (Task 2 approach - local files)
import joblib
model = joblib.load('models/best_model.joblib')

# Railway MLflow model loading (Task 3 approach - remote server)
import mlflow
mlflow.set_tracking_uri("https://mlflow-tracking-production-53fb.up.railway.app")

# Load model from Railway MLflow server
model = mlflow.sklearn.load_model("runs:/{run_id}/model")

# Or load from model registry (if registered)
model = mlflow.sklearn.load_model("models:/heart_disease_random_forest/1")
```

### **When to Use Which**

- **Use `models/` files**: Quick prototyping, Jupyter notebooks, offline development
- **Use Railway MLflow models**: Production deployment, team collaboration, experiment comparison
- **Use Task 4 packages**: Complete deployment with all dependencies and configurations

### **Task 4: Model Packaging & Reproducibility (Completed)**

**File**: `src/model_packaging.py`

**Implementation completed:**

- Production-ready model packaging system
- Multi-format model exports (joblib, pickle, MLflow)
- Complete preprocessing pipeline packaging
- Environment snapshot and dependency management
- Reproducibility validation and testing
- Deployment-ready package creation

**Key Features:**

- **Champion Model Identification**: Automatically identifies best performing model from MLflow registry
- **Multi-Format Export**: Exports models in joblib, pickle, and MLflow formats for different deployment scenarios
- **Complete Preprocessing Pipeline**: Packages entire preprocessing pipeline with missing value handling and feature engineering
- **Environment Snapshots**: Captures exact package versions, conda environment, and system information
- **Configuration Management**: Creates deployment and model configuration files
- **Package Validation**: Comprehensive testing of packaged models and components
- **Production-Ready Structure**: Organized package structure with README, metadata, and deployment guides

**Package Contents:**

- `model.joblib` - Complete pipeline (preprocessing + model)
- `model.pkl` - Complete pipeline in pickle format
- `mlflow_model/` - MLflow model format with metadata
- `preprocessing_pipeline.joblib` - Preprocessing pipeline only
- `model_only.joblib` - Trained model without preprocessing
- `model_config.yaml` - Model configuration
- `deployment_config.yaml` - Deployment configuration
- `model_metadata.json` - Complete model metadata
- `requirements_exact.txt` - Exact package versions
- `requirements_core.txt` - Minimal deployment requirements
- `environment.yml` - Conda environment specification
- `system_info.json` - System information
- `README.md` - Package documentation

**To run Task 4:**

```bash
conda activate myenv
python src/model_packaging.py
```

**Outputs:**

- `packages/{timestamp}/` - Complete deployment package
- `environments/` - Environment snapshots
- `configs/` - Configuration files
- Validated and tested model packages ready for production deployment

**Model Storage Integration:**
Task 4 unifies both Task 2 (local storage) and Task 3 (Railway MLflow storage) approaches by creating comprehensive deployment packages that include models from both sources, complete preprocessing pipelines, and all necessary metadata for production deployment. The Railway integration ensures team collaboration and professional experiment tracking.

## Git Configuration

The project includes a comprehensive `.gitignore` file that excludes:

- Generated data files (keeps directory structure)
- Trained models (keeps directory structure)
- Python cache files and virtual environments
- IDE and OS specific files
- Temporary and backup files

**Note**: MLflow tracking is handled by Railway server (remote), so no local MLflow artifacts or database files need to be managed in Git.

Important directories are preserved with `.gitkeep` files.

## Implementation Steps

### Task 1: Data Acquisition & Exploratory Data Analysis

**File**: `src/data_acquisition_eda.py`

**Implementation completed:**

- Dataset loading from UCI ML Repository with fallback to local files
- Data quality assessment (missing values, duplicates, data types)
- Comprehensive EDA with statistical analysis
- Visualization functions (commented for Jupyter conversion)
- Data saving and quality reporting

**Key Features:**

- Automatic dataset acquisition from UCI ML Repository
- Comprehensive data quality checks
- Statistical analysis by target variable
- Professional visualization templates (ready for Jupyter)
- Structured data saving for downstream tasks

**To run Task 1:**

```bash
cd src
python data_acquisition_eda.py
```

**Outputs:**

- `data/raw/heart_disease_raw.csv` - Raw dataset
- `data/processed/features.csv` - Feature matrix
- `data/processed/target.csv` - Target variable
- `data/processed/data_quality_report.json` - Quality assessment
- `figures/` - EDA visualizations (when uncommented)

**For Jupyter Notebook conversion:**

1. Copy the code to a new notebook
2. Uncomment all matplotlib/seaborn visualization code
3. Run cells individually for interactive analysis

### Task 2: Feature Engineering & Model Development

**File**: `src/feature_engineering.py`

**Implementation completed:**

- Missing value handling with median/mode imputation
- Binary target conversion (multi-class → binary classification)
- Advanced feature engineering (6 new features created)
- Multiple model training (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- Cross-validation with stratified K-fold
- Hyperparameter tuning with GridSearchCV
- Comprehensive model evaluation and comparison
- Model persistence and results saving

**Key Features:**

- Robust preprocessing pipeline with RobustScaler
- Engineered features: age groups, metabolic ratios, interaction terms
- 4 different algorithms with hyperparameter optimization
- Comprehensive evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Visualization templates ready for Jupyter conversion

**Model Performance Results:**

- **Best Model**: Logistic Regression (ROC-AUC: 0.9610)
- **Runner-up**: Gradient Boosting (ROC-AUC: 0.9481)
- **Third**: SVM (ROC-AUC: 0.9470)
- All models achieved >85% accuracy on test set

**To run Task 2:**

```bash
cd src
python feature_engineering.py
```

**Outputs:**

- `models/best_model.joblib` - Best performing model (Task 2 direct save)
- `models/{model_name}_model.joblib` - Individual trained models (Task 2 direct save)
- `models/evaluation_results.json` - Performance metrics and comparison
- `models/feature_names.json` - Feature list for reproducibility
- Visualization code ready for Jupyter notebooks

**Model Storage Explanation:**
Task 2 saves models directly to `models/` directory using joblib format for quick access and backup. These are standalone model files without MLflow integration.

### Task 3: MLflow Experiment Tracking (Railway Integration)

**File**: `src/experiment_tracking.py`

**Implementation completed:**

- **Railway MLflow server integration** (shared team access)
- **Model artifacts storage** on Railway server (MLflow 3.8.0 ↔ 2.10.0 compatibility)
- **Comprehensive experiment logging** (parameters, metrics, artifacts)
- **Model registration and versioning** with Railway compatibility
- **Cross-validation results tracking** and performance comparison
- **Automated model comparison and ranking** across algorithms
- **Experiment summary and analysis** with team collaboration
- **Integration with Tasks 1 and 2** for complete pipeline tracking

**Key Features:**

- **Railway MLflow Server**: `https://mlflow-tracking-production-53fb.up.railway.app`
- **Team Collaboration**: Shared experiments and model registry accessible to all teammates
- **Model Storage**: Both experiment metadata AND actual model files stored on Railway
- **Compatibility Solution**: MLflow 3.8.0 client works with Railway 2.10.0 server
- **Complete Tracking**: Parameters, metrics, artifacts, cross-validation, model comparison
- **Professional Interface**: Web-based MLflow UI for experiment analysis and visualization

**Railway Integration Benefits:**

- No local database files or synchronization issues
- Shared access for entire team without conflicts
- Professional experiment tracking interface
- Model versioning and registry with remote storage
- Complete experiment reproducibility and collaboration

**MLflow Components Tracked:**

- **Parameters**: Model hyperparameters, dataset info, feature engineering settings
- **Metrics**: Accuracy, precision, recall, F1-score, ROC-AUC, cross-validation scores
- **Artifacts**: Trained models, classification reports, confusion matrices
- **Tags**: Experiment stage, task number, performance tier, model family

**To run Task 3:**

```bash
conda activate myenv
python src/experiment_tracking.py
```

**To view MLflow UI:**

```bash
# Open Railway MLflow Server (accessible to all teammates)
# URL: https://mlflow-tracking-production-53fb.up.railway.app
```

**Outputs:**

- **Railway server experiment tracking** (remote, shared)
- **Model artifacts stored on Railway** (complete models with metadata)
- **Registered models in MLflow Model Registry** with versioning
- **Experiment comparison and analysis** (web-based interface)
- **Model performance rankings** and automated comparison
- **Shared team access** to all experiments and models (no local conflicts)

**Model Storage Explanation:**
Task 3 stores both experiment metadata AND actual model files on the Railway MLflow server. This provides complete model lifecycle management with team collaboration, versioning, and professional experiment tracking interface. The compatibility solution ensures MLflow 3.8.0 clients work seamlessly with Railway 2.10.0 server.

### All Tasks Completed

All four main tasks of the MLOps pipeline have been successfully implemented:

- **Task 1**: Data Acquisition & Exploratory Data Analysis ✓
- **Task 2**: Feature Engineering & Model Development ✓
- **Task 3**: MLflow Experiment Tracking ✓
- **Task 4**: Model Packaging & Reproducibility ✓

The project now provides a complete end-to-end MLOps pipeline for heart disease prediction, from data acquisition to production-ready model packaging.

- `figures/` - EDA visualizations (when uncommented)

**For Jupyter Notebook conversion:**

1. Copy the code to a new notebook
2. Uncomment all matplotlib/seaborn visualization code
3. Run cells individually for interactive analysis

## Dataset Information

**Source**: UCI Machine Learning Repository - Heart Disease Dataset
**Features**: 13 clinical features (age, sex, chest pain type, blood pressure, etc.)
**Target**: Binary classification (presence/absence of heart disease)
**Samples**: ~303 instances

## Notes

- All visualization code is commented in Python files for easy Jupyter conversion
- The project follows MLOps best practices with modular, reproducible code
- Fallback mechanisms ensure dataset loading works in different environments
- Comprehensive logging and error handling throughout
- **Railway MLflow integration** provides professional experiment tracking with team collaboration
- **Model storage compatibility** ensures MLflow 3.8.0 clients work with Railway 2.10.0 server
- **No local database management** required - all tracking handled by Railway server
