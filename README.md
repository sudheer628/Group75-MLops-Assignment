# Heart Disease Prediction MLOps Project

[![CI Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/CI%20Pipeline%20-%20Lint%20and%20Test/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions)
[![Model Training](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/Model%20Training%20Pipeline/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-3.8.0-orange.svg)](https://mlflow.org/)
[![Railway](https://img.shields.io/badge/Railway-MLflow%20Server-green.svg)](https://railway.app/)

This project implements an end-to-end MLOps pipeline for heart disease prediction using the UCI Heart Disease dataset with professional CI/CD practices.

## Project Structure

```
├── app/                          # FastAPI application
│   ├── main.py                   # FastAPI server and endpoints
│   ├── models.py                 # Pydantic models for validation
│   ├── prediction.py             # ML prediction logic
│   └── config.py                 # Configuration settings
├── src/                          # Source code
│   ├── data_acquisition_eda.py   # Task 1: Data acquisition and EDA
│   ├── feature_engineering.py    # Task 2: Feature engineering and model development
│   ├── experiment_tracking.py    # Task 3: MLflow experiment tracking
│   ├── model_packaging.py        # Task 4: Model packaging and reproducibility
│   └── ci_utils.py               # CI/CD utilities
├── test-data/                    # API test data
│   ├── sample-input.json         # High-risk test case
│   └── sample-input-healthy.json # Low-risk test case
├── scripts/                      # Testing and utility scripts
│   └── test-api-cloud.sh         # Cloud-based API testing
├── .devcontainer/                # GitHub Codespaces configuration
│   └── devcontainer.json         # VS Code dev container setup
├── .github/workflows/            # GitHub Actions CI/CD
│   ├── ci.yml                    # Main CI pipeline
│   ├── container-build.yml       # Container build and registry
│   ├── model-training.yml        # Model training pipeline
│   └── pr-validation.yml         # PR validation
├── data/                         # Data storage (gitignored except structure)
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed datasets
├── models/                       # Trained models (gitignored except structure)
├── tests/                        # Unit tests
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Container orchestration
├── requirements.txt              # Python dependencies (ML pipeline)
├── requirements-api.txt          # API-specific dependencies
├── CODESPACES.md                 # GitHub Codespaces guide
└── README.md                     # This file
```

## CI/CD Pipeline (TASK-5)

### GitHub Actions Workflows

The project includes comprehensive CI/CD pipelines with automated testing, linting, and model training:

#### 1. CI Pipeline (`ci.yml`)

- **Triggers**: Push to main/develop, Pull Requests
- **Python Versions**: 3.11, 3.12
- **Features**:
  - Code linting with flake8
  - Code formatting with black
  - Import sorting with isort
  - Comprehensive test suite
  - MLflow integration testing
  - Artifact generation

#### 2. Model Training Pipeline (`model-training.yml`)

- **Triggers**: Manual dispatch, Weekly schedule
- **Features**:
  - Complete pipeline execution
  - MLflow experiment tracking
  - Model packaging
  - Deployment artifact generation
  - Training reports and summaries

#### 3. PR Validation (`pr-validation.yml`)

- **Triggers**: Pull Request events
- **Features**:
  - Quick validation checks
  - Security scanning
  - Critical path testing

### CI/CD Features

- **Multi-Python Support**: Tests on Python 3.11 and 3.12
- **Dependency Caching**: Optimized build times
- **Artifact Management**: 30-90 day retention
- **Professional Reporting**: Detailed summaries and logs
- **Railway Integration**: Seamless MLflow connectivity
- **Security Scanning**: Automated code security checks

### Running CI/CD Locally

```bash
# Install CI/CD tools
pip install black isort flake8

# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check code quality
flake8 src/ tests/

# Run tests in CI mode
python run_tests.py --ci
```

### GitHub Secrets Required

Configure in `Settings → Secrets and variables → Actions`:

- `MLFLOW_TRACKING_URI`: `https://mlflow-tracking-production-53fb.up.railway.app`

## 🚀 **Quick Start (Cloud Development)**

### **Option 1: GitHub Actions Testing (Recommended)**

1. **Push changes** to GitHub
2. **Check Actions tab** for automated container builds and testing
3. **View results** in workflow summaries

### **Option 2: GitHub Codespaces Development**

1. **Click "Code" → "Codespaces" → "Create codespace"**
2. **Wait for automatic setup** (Python, Docker, dependencies)
3. **Run tests**: `chmod +x scripts/test-api-cloud.sh && ./scripts/test-api-cloud.sh`
4. **Access API** via forwarded ports

### **Option 3: Use Pre-built Container**

```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/your-username/your-repo/heart-disease-api:latest
docker run -p 8000:8000 ghcr.io/your-username/your-repo/heart-disease-api:latest
```

**📖 Detailed Guide**: See [CODESPACES.md](CODESPACES.md) for comprehensive instructions.

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

- **Use `models/` files**: Quick prototyping, offline development
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

### **Task 5: CI/CD Pipeline & Automated Testing (Completed)**

**Implementation completed:**

- **GitHub Actions workflows** (3 workflows)
- **Automated linting** (flake8, black, isort)
- **Multi-Python testing** (3.11, 3.12)
- **MLflow integration testing**
- **Artifact management** (30-90 day retention)
- **Professional reporting** and summaries
- **Security scanning** and validation

**Workflows Created:**

#### 1. **CI Pipeline** (`.github/workflows/ci.yml`)

- **Triggers**: Push to main/develop, Pull Requests
- **Features**: Code quality checks, comprehensive testing, artifact generation, container testing
- **Python Versions**: 3.11, 3.12 (matrix strategy)
- **Caching**: Optimized pip dependency caching

#### 2. **Model Training Pipeline** (`.github/workflows/model-training.yml`)

- **Triggers**: Manual dispatch, Weekly schedule (Monday 2 AM)
- **Features**: Complete pipeline execution, MLflow tracking, deployment packaging
- **Artifacts**: 90-day retention for training outputs

#### 3. **PR Validation** (`.github/workflows/pr-validation.yml`)

- **Triggers**: Pull Request events
- **Features**: Quick validation, security scanning, critical path testing

### **Task 6: Model Containerization (Completed)** 🐳

**Implementation completed:**

- **FastAPI application** with comprehensive API endpoints
- **Docker containerization** with multi-stage optimization
- **GitHub Container Registry integration** (automated)
- **Local development environment** with Docker Compose
- **Comprehensive testing** (local and CI/CD)
- **Production-ready deployment** configuration

#### **FastAPI Application Features:**

**Files**: `app/main.py`, `app/models.py`, `app/prediction.py`, `app/config.py`

- **RESTful API** with FastAPI framework
- **Input validation** with Pydantic models
- **Comprehensive endpoints**:
  - `POST /predict` - Heart disease prediction with confidence scores
  - `GET /health` - Health check and service status
  - `GET /` - API information and navigation
  - `GET /model/info` - Model metadata and information
  - `GET /docs` - Interactive API documentation (Swagger UI)
  - `GET /redoc` - Alternative API documentation

**Key Features:**

- **Robust error handling** with proper HTTP status codes
- **Input validation** with field constraints and type checking
- **Confidence scoring** with risk level assessment (Low/Medium/High)
- **Logging and monitoring** with structured logging
- **CORS support** for web applications
- **Health checks** for container orchestration
- **Security** with non-root user execution

#### **Container Features:**

**Files**: `Dockerfile`, `docker-compose.yml`, `.dockerignore`

- **Optimized Docker image** with Python 3.11 slim base
- **Multi-architecture support** (AMD64, ARM64)
- **Layer caching** for faster builds
- **Security hardening** with non-root user
- **Health checks** built into container
- **Development environment** with Docker Compose

#### **GitHub Container Registry Integration:**

**File**: `.github/workflows/container-build.yml`

- **Automated builds** on push to main/develop branches
- **Multi-platform builds** (linux/amd64, linux/arm64)
- **Automatic tagging** with git SHA, branch, and semantic versioning
- **Container testing** in CI pipeline
- **Registry push** to GitHub Container Registry (ghcr.io)
- **Artifact caching** for faster subsequent builds

#### **Cloud Development & Testing:**

**Files**: `scripts/test-api-cloud.sh`, `test-data/`, `.devcontainer/`, `CODESPACES.md`

- **GitHub Codespaces support** with automatic environment setup
- **Cloud-based testing** script for container validation
- **Sample test data** for different risk scenarios
- **Automated validation** of API responses
- **VS Code development environment** with pre-configured extensions

#### **API Usage Examples:**

**Health Check:**

```bash
curl https://your-api-url/health
```

**Prediction (High Risk):**

```bash
curl -X POST "https://your-api-url/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 3, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
  }'
```

**Response:**

```json
{
  "prediction": 1,
  "confidence": 0.85,
  "probabilities": [0.15, 0.85],
  "risk_level": "High"
}
```

#### **Container Registry:**

Images are automatically built and pushed to:

- `ghcr.io/username/repo/heart-disease-api:latest`
- `ghcr.io/username/repo/heart-disease-api:main-sha123`
- `ghcr.io/username/repo/heart-disease-api:v1.0.0` (on tags)

#### **Development Options:**

**Option 1: GitHub Actions (Recommended)**

```bash
# Simply push your changes - GitHub Actions will:
git add .
git commit -m "Update API"
git push

# Then check the Actions tab for:
# - Automated container builds
# - Comprehensive API testing
# - Container registry deployment
```

**Option 2: GitHub Codespaces**

```bash
# Open repository in Codespaces (see CODESPACES.md)
# Docker and all dependencies pre-installed

# Test the API
chmod +x scripts/test-api-cloud.sh
./scripts/test-api-cloud.sh

# Access API documentation at forwarded port
```

**Option 3: Pull from Container Registry**

```bash
# Pull and run the latest built image
docker pull ghcr.io/username/repo/heart-disease-api:latest
docker run -p 8000:8000 ghcr.io/username/repo/heart-disease-api:latest
```

#### **Production Deployment Ready:**

- **Container orchestration** ready (Kubernetes, Docker Swarm)
- **Load balancer** compatible with health checks
- **Environment configuration** through environment variables
- **Monitoring** integration points built-in
- **Scalability** designed for horizontal scaling

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
- Visualization functions
- Data saving and quality reporting

**Key Features:**

- Automatic dataset acquisition from UCI ML Repository
- Comprehensive data quality checks
- Statistical analysis by target variable
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

All five main tasks of the MLOps pipeline have been successfully implemented:

- **Task 1**: Data Acquisition & Exploratory Data Analysis ✓
- **Task 2**: Feature Engineering & Model Development ✓
- **Task 3**: MLflow Experiment Tracking ✓
- **Task 4**: Model Packaging & Reproducibility ✓
- **Task 5**: CI/CD Pipeline & Automated Testing ✓

The project now provides a complete end-to-end MLOps pipeline for heart disease prediction, from data acquisition to production-ready deployment with professional CI/CD practices.

## TASK-5: CI/CD Pipeline & Automated Testing (COMPLETED)

### Implementation Summary

**Comprehensive CI/CD pipeline implemented with:**

- **GitHub Actions workflows** (3 workflows)
- **Automated linting** (flake8, black, isort)
- **Multi-Python testing** (3.11, 3.12)
- **MLflow integration testing**
- **Artifact management** (30-90 day retention)
- **Professional reporting** and summaries
- **Security scanning** and validation

### Workflows Created

#### 1. **CI Pipeline** (`.github/workflows/ci.yml`)

- **Triggers**: Push to main/develop, Pull Requests
- **Features**: Code quality checks, comprehensive testing, artifact generation
- **Python Versions**: 3.11, 3.12 (matrix strategy)
- **Caching**: Optimized pip dependency caching

#### 2. **Model Training Pipeline** (`.github/workflows/model-training.yml`)

- **Triggers**: Manual dispatch, Weekly schedule (Monday 2 AM)
- **Features**: Complete pipeline execution, MLflow tracking, deployment packaging
- **Artifacts**: 90-day retention for training outputs

#### 3. **PR Validation** (`.github/workflows/pr-validation.yml`)

- **Triggers**: Pull Request events
- **Features**: Quick validation, security scanning, critical path testing

### Code Enhancements

#### **CI/CD Utilities** (`src/ci_utils.py`)

- Environment setup and validation
- Professional logging and reporting
- CI context detection and configuration
- Comprehensive validation functions

#### **Enhanced Test Runner** (`run_tests.py`)

- CI mode with enhanced logging (`--ci` flag)
- Professional step tracking and reporting
- Artifact generation for CI/CD workflows

#### **Configuration Files**

- `.flake8`: Code quality standards
- `pyproject.toml`: Black and isort configuration
- Updated `requirements.txt`: Added CI/CD tools

### Validation and Testing

**Comprehensive validation system:**

- **Workflow validation**: YAML syntax and structure
- **Configuration validation**: All config files present
- **CI utilities validation**: Import and functionality tests
- **Test runner validation**: CI mode implementation
- **Dependencies validation**: All required packages
- **Directory structure validation**: Required directories exist

**Test Results:**

```
GitHub Actions Workflows       PASS
Configuration Files            PASS
CI/CD Utilities                PASS
Test Runner CI Mode            PASS
Dependencies                   PASS
Directory Structure            PASS
```

### Professional Features

#### **Multi-Python Support**

- Tests on Python 3.11 and 3.12
- Matrix strategy for comprehensive compatibility testing
- Future-proof with latest Python versions

#### **Code Quality Enforcement**

- **flake8**: Python code standards and syntax checking
- **black**: Consistent code formatting (127 char line length)
- **isort**: Organized import sorting
- **Security scanning**: GitHub Super Linter integration

#### **Performance Optimizations**

- **Dependency caching**: Faster build times with pip cache
- **Conditional execution**: Skip steps on failures
- **Parallel execution**: Matrix strategy for multiple Python versions
- **Smart triggers**: Different workflows for different events

#### **Professional Reporting**

- **GitHub Step Summaries**: Rich markdown reports
- **Artifact management**: Organized file retention
- **Detailed logging**: Timestamped step execution
- **Failure notifications**: Clear error reporting

### Integration with MLOps Pipeline

**Seamless integration:**

- **Railway MLflow**: CI/CD workflows connect to Railway server
- **Environment variables**: Secure secret management
- **Artifact flow**: Test results → Training artifacts → Deployment packages
- **Validation chain**: Code quality → Tests → Training → Packaging

### Usage Instructions

#### **Automatic Execution**

- **CI Pipeline**: Runs on every push/PR automatically
- **Model Training**: Weekly schedule + manual dispatch
- **PR Validation**: Automatic on pull request events

#### **Manual Triggers**

```bash
# Local CI testing
python run_tests.py --ci

# Local code quality
black src/ tests/
isort src/ tests/
flake8 src/ tests/

# Validation
python validate_cicd.py
```

#### **GitHub Secrets Required**

Configure in repository settings:

- `MLFLOW_TRACKING_URI`: `https://mlflow-tracking-production-53fb.up.railway.app`

### Production Readiness

**Enterprise-grade features:**

- **Multi-environment testing**
- **Automated quality gates**
- **Security scanning**
- **Artifact management**
- **Professional reporting**
- **Failure handling**
- **Performance optimization**

**Compliance and Standards:**

- **Code quality standards** (PEP 8 compliance)
- **Security best practices** (secret management)
- **Documentation standards** (comprehensive README)
- **Testing standards** (comprehensive test coverage)

The CI/CD pipeline is now production-ready and follows industry best practices for MLOps workflows.

- `figures/` - EDA visualizations (when uncommented)

## Dataset Information

**Source**: UCI Machine Learning Repository - Heart Disease Dataset
**Features**: 13 clinical features (age, sex, chest pain type, blood pressure, etc.)
**Target**: Binary classification (presence/absence of heart disease)
**Samples**: ~303 instances

## Notes

- The project follows MLOps best practices with modular, reproducible code
- Fallback mechanisms ensure dataset loading works in different environments
- Comprehensive logging and error handling throughout
- **Railway MLflow integration** provides professional experiment tracking with team collaboration
- **Model storage compatibility** ensures MLflow 3.8.0 clients work with Railway 2.10.0 server
- **No local database management** required - all tracking handled by Railway server
