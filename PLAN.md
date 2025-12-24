# Assignment-1 Implementation Plan

This plan maps the assignment tasks (from `INSTRUCTIONS.md`) to recommended tools, platforms, and actionable steps. We can run the solution locally (Docker Desktop / Minikube) or in EKS (AWS Account required).

---

## Overview

---

## Tasks (mapped)

Task 1 — Data Acquisition & EDA (5 marks)

- Tools: Python (pandas, matplotlib/seaborn/plotly), Jupyter Notebook
- Storage: local CSV or `dvc` tracked dataset with S3/GCS remote
- Local run: Jupyter / `notebooks/EDA.ipynb`
- Notes: include `import_dataset.py` and a `data/README.md` with download steps

Task 2 — Feature Engineering & Model Development (8 marks)

- Tools: scikit-learn, pandas, joblib/ONNX (optional), Jupyter/py scripts
- Tracking: use MLflow to log experiments
- Local run: `train.py` with flags for quick dev vs full run

Task 3 — Experiment Tracking (5 marks)

- Tools: MLflow (server or local UI), backend store: SQLite for local, artifact store: S3 (or local `mlruns/` for teammates)
- Options: local MLflow UI for teammates; remote S3-backed MLflow for full pipeline runs

Task 4 — Model Packaging & Reproducibility (7 marks)

- Tools: MLflow model registry or `joblib`/`pickle` / ONNX for portability
- Environment: `requirements.txt` (or `environment.yml`) + `Dockerfile`
- Repro: provide preprocessing pipeline (scikit-learn `Pipeline`) saved with the model

Task 5 — CI/CD & Automated Testing (8 marks)

- Primary: **GitHub Actions** workflows
  - Jobs: `lint`, `unit-tests`, `build-image`, `dvc-push` (optional), `train-and-register` (optional), `deploy` (optional)
- Tests: `pytest` for data checks, model unit tests (expected shapes, small sample inference)

Task 6 — Model Containerization (5 marks)

- Tools: Docker, `Dockerfile` for model server (FastAPI recommended)
- Local options:
  - Docker Desktop: `docker build` + `docker run`
  - Docker Compose: run app + optional local Prometheus/Grafana
- Kubernetes option: provide manifests/Helm chart

Task 7 — Production Deployment (7 marks)

- Option 1 (local, teammate-friendly): **Minikube / Docker Desktop Kubernetes**
  - Use Kustomize overlays or a Helm chart `overlays/minikube`
  - Ingress: Minikube Ingress addon
  - Simple stack: deploy model service + optional metrics exporter
- Option 2 (AWS): **EKS**
  - Provision with `Terraform` (or eksctl for quick setup)
  - Use Helm for `kube-prometheus-stack` and model-serving tools (Seldon/KServe optional)
  - Production ingress: AWS ALB or NLB via Ingress Controller

Task 8 — Monitoring & Logging (3 marks)

- Tools (local friendly): Prometheus + Grafana via Helm or Docker Compose
- EKS: `kube-prometheus-stack` (Prometheus Operator) + Grafana; logs to CloudWatch (optional)
- Simple model-drift: Evidently or simple custom metrics logged to MLflow/Prometheus

Task 9 — Documentation & Reporting (2 marks)

- Deliver: `README.md`, `screenshot/` folder, `report.docx` (10 pages), short demo video
- Include an architecture diagram (PNG/SVG) showing local vs EKS paths

---

## Recommended repository layout

- `Dockerfile` — model server image
- `docker-compose.yml` — local dev stack (app + prometheus + grafana)
- `notebooks/` — EDA + modelling
- `src/` — training, inference, preprocessing modules
- `tests/` — unit/integration tests
- `dvc.yaml` + `.dvc/` — data pipeline and tracked data pointers
- `mlflow/` or `mlruns/` — local mlflow artifacts
- `k8s/` — manifests or Helm chart
  - `k8s/overlays/minikube/`
  - `k8s/overlays/eks/`
- `terraform/` — optional: infra provisioning for EKS + S3
- `README.md`, `PLAN.md`, `report.docx`, `screenshot/`

---

## Deployment options & commands (concise)

Local (Docker Desktop)

```bash
docker build -t heart-model:dev .
docker run -p 8000:8000 heart-model:dev
# Or using compose:
docker-compose up -d
```

Minikube (team-friendly)

```bash
minikube start --driver=docker --cpus=4 --memory=8192
kubectl apply -k k8s/overlays/minikube
minikube addons enable ingress
kubectl port-forward svc/model-service 8000:80
```

EKS (optional)

```bash
# Provision infra (Terraform or eksctl)
terraform init && terraform apply
# Build and push image to registry
docker build -t <registry>/heart-model:latest .
docker push <registry>/heart-model:latest
# Deploy via Helm/kubectl
kubectl apply -k k8s/overlays/eks
```

Install Prometheus + Grafana (Minikube or EKS using Helm)

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install kube-prom-stack prometheus-community/kube-prometheus-stack --namespace monitoring --create-namespace
kubectl port-forward -n monitoring svc/kube-prom-stack-grafana 3000:80
```

---

## CI/CD (GitHub Actions)

- GitHub Actions workflows to include:

  - `ci.yml` — lint, unit-tests, build Docker image, push to GitHub Packages or Docker Hub
  - `train.yml` — (optional) run training job, DVC push, MLflow log
  - `deploy-minikube.yml` — workflow that builds image and prints dev instructions (teammate-friendly)
  - `deploy-eks.yml` — workflow that applies `k8s/overlays/eks` using kubeconfig stored in GitHub Secrets

Secrets to store in CI:

- Container registry creds, AWS credentials (for EKS + S3), MLflow artifact S3 creds, kubeconfig (or use OIDC for EKS), DVC remote access keys

---

## Minimal onboarding checklist for teammates

1. Install Docker Desktop (enable Kubernetes) or Minikube.
2. Clone repo and run `pip install -r requirements.txt`.
3. For local: `docker-compose up -d` then test `http://localhost:8000/predict`.
4. For minikube: run provided `minikube start` then `kubectl apply -k k8s/overlays/minikube`.
5. Use MLflow UI locally: `mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns`.

---

## Optional Enhancements (Advanced Features)

### Data Validation

- **Great Expectations**: Add data quality checks and validation suites
  - Create expectation suites for input data schema validation
  - Integrate with CI/CD to fail on data quality issues
- **Pydantic Schemas**: Simple input validation for API endpoints

  ```python
  # src/schemas.py
  from pydantic import BaseModel, Field

  class PredictionInput(BaseModel):
      age: int = Field(..., ge=0, le=120)
      sex: int = Field(..., ge=0, le=1)
      cp: int = Field(..., ge=0, le=3)
      # ... other features with validation rules
  ```

### Model Versioning Strategy

- **Blue-Green Deployment**: Maintain two identical production environments
  - Deploy new model version to "green" environment
  - Switch traffic after validation
  - Keep "blue" as rollback option
- **Canary Deployment**: Gradual rollout strategy
  - Route small percentage of traffic to new model version
  - Monitor performance metrics and gradually increase traffic
  - Implement using Kubernetes deployments with traffic splitting
- **A/B Testing**: Compare model performance with live traffic
  ```yaml
  # k8s/overlays/eks/canary-deployment.yaml
  apiVersion: argoproj.io/v1alpha1
  kind: Rollout
  metadata:
    name: model-rollout
  spec:
    strategy:
      canary:
        steps:
          - setWeight: 10
          - pause: { duration: 10m }
          - setWeight: 50
          - pause: { duration: 10m }
  ```

### Testing Strategy Enhancement

- **Integration Tests**: End-to-end API testing

  ```python
  # tests/test_integration.py
  def test_prediction_endpoint():
      response = client.post("/predict", json=sample_input)
      assert response.status_code == 200
      assert "prediction" in response.json()

  def test_health_endpoint():
      response = client.get("/health")
      assert response.status_code == 200
  ```

- **Load Testing**: Use `locust` or `k6` for performance validation
- **Model Performance Tests**: Validate model accuracy on test datasets
- **Contract Testing**: Ensure API schema compatibility across versions

### Security Enhancements

- **API Authentication**: JWT tokens or API keys
- **Rate Limiting**: Prevent API abuse
- **Input Sanitization**: Additional validation layers
- **HTTPS/TLS**: Secure communication in production

## Notes & trade-offs

- KServe/Seldon: production-grade but increases complexity — include as an optional `k8s/overlays/eks/advanced` overlay. For this assignment, stick with basic Kubernetes deployments. We can add a simple note in documentation about production alternatives without implementing them.
- Use DVC + S3 for data persistency in CI;

```

  # Add to plan:
  data/
  ├── raw/           # Local copies for quick start
  ├── processed/     # DVC-tracked processed data
  └── download.py    # Script using import_dataset.py

```

This way teammates can start immediately with local data, but CI/CD uses DVC for reproducibility.

- MLflow Backend Complexity → Progressive Setup
  Start with SQLite locally, add S3 backend as optional

```

  # docker-compose.yml - add MLflow service
  mlflow:
    image: python:3.9
    command: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0
    ports:
      - "5000:5000"

```

- Monitoring Setup → Simplified Stack
  Consider adding a "monitoring-lite" option using just FastAPI's built-in metrics + simple logging:

```

  # In FastAPI app
  import logging
  from prometheus_client import Counter, Histogram

  prediction_counter = Counter('predictions_total', 'Total predictions')
  prediction_latency = Histogram('prediction_duration_seconds', 'Prediction latency')

```
