# Heart Disease Prediction - MLOps Pipeline

This is our MLOps course assignment where we built an end-to-end machine learning pipeline for predicting heart disease. We went through the complete lifecycle - from getting the data, training models, tracking experiments, containerizing the application, setting up CI/CD, deploying to cloud, and adding monitoring. It was a good learning experience putting all these pieces together.

## Project Overview

We built a binary classifier that predicts whether a patient has heart disease or not, using the UCI Heart Disease dataset. The model is served through a REST API that we deployed on a GCP VM, and we set up Grafana Cloud to monitor everything. The whole pipeline is automated with GitHub Actions.

### Key Features

- Automated ML pipeline from data acquisition to deployment
- Experiment tracking with MLflow (Railway-hosted)
- Containerized FastAPI application
- CI/CD with GitHub Actions
- Production deployment on GCP VM
- Monitoring and logging with Grafana Cloud

### Live Deployment

- API Endpoint: http://myprojectdemo.online
- API Documentation: http://myprojectdemo.online/docs
- MLflow Experiment Tracking: https://mlflow-tracking-production-53fb.up.railway.app/
- Grafana Metrics Dashboard: https://group75mlops.grafana.net/d/suwdlv9/group75-assignment

---

## Architecture

```
+------------------+     +------------------+     +------------------+
|   Data Source    |     |  Model Training  |     |   Experiment     |
|   (UCI Dataset)  | --> |  (scikit-learn)  | --> |   Tracking       |
|                  |     |                  |     |   (MLflow)       |
+------------------+     +------------------+     +------------------+
                                                          |
                                                          v
+------------------+     +------------------+     +------------------+
|   Monitoring     |     |   Production     |     |   Container      |
|   (Grafana)      | <-- |   (GCP VM)       | <-- |   (Docker)       |
+------------------+     +------------------+     +------------------+
```

### Technology Stack

| Component           | Technology                        |
| ------------------- | --------------------------------- |
| ML Framework        | scikit-learn                      |
| API Framework       | FastAPI                           |
| Experiment Tracking | MLflow (Railway)                  |
| Containerization    | Docker                            |
| Container Registry  | GitHub Container Registry (GHCR)  |
| CI/CD               | GitHub Actions                    |
| Cloud Platform      | Google Cloud Platform (GCP)       |
| Reverse Proxy       | Nginx                             |
| Monitoring          | Grafana Cloud (Prometheus + Loki) |

---

## Project Structure

```
.
├── app/                              # FastAPI application
│   ├── main.py                       # API endpoints
│   ├── models.py                     # Pydantic schemas
│   ├── prediction.py                 # ML prediction logic
│   ├── metrics.py                    # Prometheus metrics
│   └── config.py                     # Configuration
├── src/                              # ML pipeline source code
│   ├── data_acquisition_eda.py       # Task 1: Data & EDA
│   ├── feature_engineering.py        # Task 2: Feature engineering
│   ├── experiment_tracking.py        # Task 3: MLflow tracking
│   └── model_packaging.py            # Task 4: Model packaging
├── tests/                            # Unit tests
├── models/                           # Trained model artifacts
├── data/                             # Dataset storage
│   ├── raw/                          # Raw data
│   └── processed/                    # Processed data
├── .github/workflows/                # CI/CD pipelines
│   ├── ci.yml                        # Main CI pipeline
│   ├── container-build.yml           # Container build & push
│   ├── deploy.yml                    # Auto-deploy to GCP VM
│   ├── pr-validation.yml             # PR validation checks
│   └── model-training.yml            # Model training pipeline
├── Dockerfile                        # Container definition
├── docker-compose.yml                # Container orchestration
├── nginx.conf                        # Nginx configuration
├── requirements.txt                  # ML dependencies
└── requirements-api.txt              # API dependencies
```

---

## Task Implementation

### Task 1: Data Acquisition and EDA

Source: `src/data_acquisition_eda.py`

- Dataset: UCI Heart Disease (303 samples, 14 features)
- Automated download from UCI ML Repository
- Data quality assessment and cleaning
- Statistical analysis and visualization

Output files:

- `data/raw/heart_disease_raw.csv`
- `data/processed/features.csv`
- `data/processed/target.csv`
- `data/processed/data_quality_report.json`

### Task 2: Feature Engineering and Model Development

Source: `src/feature_engineering.py`

Models trained:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

Engineered features (6 new):

- age_group: Age categorization
- chol_age_ratio: Cholesterol to age ratio
- heart_rate_reserve: Max HR - (220 - age)
- risk_score: Composite risk metric
- age_sex_interaction: Age \* Sex
- cp_exang_interaction: Chest pain \* Exercise angina

Best model performance:

- Algorithm: Logistic Regression
- ROC-AUC: 0.9610
- Accuracy: >85%

Output files:

- `models/best_model.joblib`
- `models/evaluation_results.json`
- `models/feature_names.json`

### Task 3: Experiment Tracking

Source: `src/experiment_tracking.py`

MLflow Server: https://mlflow-tracking-production-53fb.up.railway.app

Tracked components:

- Parameters: Hyperparameters, dataset info
- Metrics: Accuracy, precision, recall, F1, ROC-AUC
- Artifacts: Models, reports, confusion matrices
- Model Registry: Versioned models with aliases

### Task 4: Model Packaging

Source: `src/model_packaging.py`

Package contents:

- Model files (joblib, pickle, MLflow format)
- Preprocessing pipeline
- Environment specifications
- Configuration files
- Deployment documentation

Output: `packages/{timestamp}/`

### Task 5: CI/CD Pipeline

Location: `.github/workflows/`

Workflows:

| Workflow        | File                  | Trigger                   | Purpose                       |
| --------------- | --------------------- | ------------------------- | ----------------------------- |
| CI Pipeline     | `ci.yml`              | Push to main/develop, PRs | Linting, unit tests           |
| Container Build | `container-build.yml` | After CI success          | Build & push Docker image     |
| Deploy to GCP   | `deploy.yml`          | After Container Build     | Auto-deploy to production     |
| PR Validation   | `pr-validation.yml`   | Pull Requests             | Quick validation before merge |
| Model Training  | `model-training.yml`  | Manual, Weekly (Mon 2AM)  | Full model retraining         |

Deployment flow (automatic on push to main):

```
ci.yml --> container-build.yml --> deploy.yml --> Live at myprojectdemo.online
```

### Task 6: Model Containerization

Files: `Dockerfile`, `docker-compose.yml`

Container image: `ghcr.io/sudheer628/group75-mlops-assignment/heart-disease-api`

API Endpoints:
| Endpoint | Method | Description |
|---------------|--------|--------------------------------|
| / | GET | API information |
| /health | GET | Health check |
| /predict | POST | Heart disease prediction |
| /model/info | GET | Model metadata |
| /metrics | GET | Prometheus metrics |
| /docs | GET | Swagger documentation |

### Task 7: Production Deployment

Platform: Google Cloud Platform (GCP VM)

Deployment details:

- Domain: myprojectdemo.online
- VM IP: 35.233.155.69
- Containers: heart-disease-api, nginx-proxy
- Orchestration: Docker Compose

Architecture:

```
Internet --> Domain (myprojectdemo.online)
         --> GCP VM (35.233.155.69)
         --> Nginx (Port 80)
         --> FastAPI (Port 8000)
```

### Task 8: Monitoring and Logging

Platform: Grafana Cloud (https://group75mlops.grafana.net/)

Components:

- Grafana Alloy: Metrics and logs collection agent
- Prometheus: Metrics storage and querying
- Loki: Log aggregation

Metrics collected:

- heart_disease_predictions_total
- heart_disease_prediction_latency_seconds
- api_requests_total
- api_errors_total

Log queries:

```logql
{job="docker"}                    # All container logs
{job="docker"} |= "predict"       # Prediction requests
{job="docker"} |~ "(?i)error"     # Error logs
```

---

## Quick Start

### Option 1: Purchased this domain in namecheap.com for application deployment

Application Domain: http://myprojectdemo.online/ (routed to GCP VM public IP)

Windows Command Prompt (CMD):

```cmd
curl http://myprojectdemo.online/

curl http://myprojectdemo.online/health

curl http://myprojectdemo.online/model/info

curl -X POST http://myprojectdemo.online/predict -H "Content-Type: application/json" -d "{\"age\": 55, \"sex\": 1, \"cp\": 3, \"trestbps\": 140, \"chol\": 250, \"fbs\": 0, \"restecg\": 1, \"thalach\": 150, \"exang\": 0, \"oldpeak\": 1.5, \"slope\": 2, \"ca\": 0, \"thal\": 3}"
```

Bash/Linux/Mac:

```bash
# API info
curl http://myprojectdemo.online/

# Health check
curl http://myprojectdemo.online/health

# Model info
curl http://myprojectdemo.online/model/info

# Make prediction
curl -X POST http://myprojectdemo.online/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 3, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
  }'
```

### Option 2: Run Container Locally

```bash
# Pull and run
docker pull ghcr.io/sudheer628/group75-mlops-assignment/heart-disease-api:latest
docker run -p 8000:8000 ghcr.io/sudheer628/group75-mlops-assignment/heart-disease-api:latest

# Test
curl http://localhost:8000/health
```

### Option 3: Local Development

```bash
# Clone repository
git clone https://github.com/sudheer628/group75-mlops-assignment.git
cd group75-mlops-assignment

# Create environment
conda create -n myenv python=3.9
conda activate myenv

# Install dependencies
pip install -r requirements.txt

# Run tests
python run_tests.py

# Run individual tasks
python src/data_acquisition_eda.py    # Task 1
python src/feature_engineering.py     # Task 2
python src/experiment_tracking.py     # Task 3
python src/model_packaging.py         # Task 4
```

---

## API Usage

### Input Features

| Feature  | Type  | Range   | Description                           |
| -------- | ----- | ------- | ------------------------------------- |
| age      | int   | 1-120   | Age in years                          |
| sex      | int   | 0-1     | Sex (0: female, 1: male)              |
| cp       | int   | 0-3     | Chest pain type                       |
| trestbps | int   | 50-300  | Resting blood pressure (mm Hg)        |
| chol     | int   | 100-600 | Serum cholesterol (mg/dl)             |
| fbs      | int   | 0-1     | Fasting blood sugar > 120 mg/dl       |
| restecg  | int   | 0-2     | Resting ECG results                   |
| thalach  | int   | 60-220  | Maximum heart rate achieved           |
| exang    | int   | 0-1     | Exercise induced angina               |
| oldpeak  | float | 0-10    | ST depression induced by exercise     |
| slope    | int   | 0-2     | Slope of peak exercise ST segment     |
| ca       | int   | 0-4     | Number of major vessels (fluoroscopy) |
| thal     | int   | 0-3     | Thalassemia type                      |

### Response Format

```json
{
  "prediction": 1,
  "confidence": 0.85,
  "probabilities": [0.15, 0.85],
  "risk_level": "High"
}
```

Risk levels:

- Low: confidence < 0.4
- Medium: 0.4 <= confidence < 0.7
- High: confidence >= 0.7

---

## Configuration

### Environment Variables

| Variable            | Default                                                | Description     |
| ------------------- | ------------------------------------------------------ | --------------- |
| MODEL_PATH          | models/best_model.joblib                               | Model file path |
| MLFLOW_TRACKING_URI | https://mlflow-tracking-production-53fb.up.railway.app | MLflow server   |
| HOST                | 0.0.0.0                                                | API host        |
| PORT                | 8000                                                   | API port        |
| LOG_LEVEL           | INFO                                                   | Logging level   |

### GitHub Secrets

Required for CI/CD:

- `MLFLOW_TRACKING_URI`: MLflow server URL
- `GHCR_TOKEN`: GitHub Container Registry token (auto-provided)

---

## Testing

```bash
# Run all tests
python run_tests.py

# Run specific task tests
python run_tests.py --task1
python run_tests.py --task2
python run_tests.py --task3
python run_tests.py --task4

# Validate environment only
python run_tests.py --validate-only
```

---

## Documentation

| Document             | Description                |
| -------------------- | -------------------------- |
| INSTRUCTIONS.md      | Assignment requirements    |
| PLAN.md              | Implementation plan        |
| DEPLOYMENT-PLAN.md   | GCP deployment guide       |
| MONITORING-PLAN.md   | Grafana Cloud setup        |
| PIPELINE_WORKFLOW.md | Development workflow guide |
| SETUP.md             | Quick setup instructions   |

---

## Repository

- GitHub: https://github.com/sudheer628/group75-mlops-assignment
- Container: ghcr.io/sudheer628/group75-mlops-assignment/heart-disease-api
- MLflow Dashboard: https://mlflow-tracking-production-53fb.up.railway.app/
- Grafana Dashboard: https://group75mlops.grafana.net/d/suwdlv9/group75-assignment

---

## License

This project is part of an MLOps course assignment (Group75) 2025-2026 BITS WILP
