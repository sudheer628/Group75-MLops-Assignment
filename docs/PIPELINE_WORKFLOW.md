# CI/CD Pipeline Workflow Guide

This document explains how we set up the CI/CD pipeline for our Heart Disease Prediction project. We used GitHub Actions to automate testing, building, and deploying our application.

---

## How Our CI/CD Works

We designed the pipeline so that every time we push code to the main branch, it automatically goes through testing, builds a Docker container, and deploys to our GCP VM. No manual steps needed once it's set up.

Here's the flow:

```
Push code to main
       |
       v
+------------------+
|   CI Pipeline    |  <-- Runs linting and unit tests
|    (ci.yml)      |
+--------+---------+
         |
         v (if tests pass)
+------------------+
| Container Build  |  <-- Builds Docker image, pushes to GHCR
|(container-build) |
+--------+---------+
         |
         v (if build succeeds)
+------------------+
|  Deploy to GCP   |  <-- SSHs into VM, pulls new image, restarts containers
|   (deploy.yml)   |
+--------+---------+
         |
         v
   Live at myprojectdemo.online
```

The whole process takes about 6-10 minutes from push to production.

---

## Our GitHub Actions Workflows

We have 5 workflow files in `.github/workflows/`. Here's what each one does:

| Workflow        | File                | When it Runs               | What it Does                    |
| --------------- | ------------------- | -------------------------- | ------------------------------- |
| CI Pipeline     | ci.yml              | Every push to main/develop | Linting, unit tests, SonarCloud |
| Container Build | container-build.yml | After CI passes            | Builds and pushes Docker image  |
| Deploy to GCP   | deploy.yml          | After container build      | Deploys to production VM        |
| PR Validation   | pr-validation.yml   | On pull requests           | Quick checks before merge       |
| Model Training  | model-training.yml  | Manual or weekly           | Retrains ML models              |

---

## Workflow Details

### 1. CI Pipeline (ci.yml)

This is the first workflow that runs when we push code. It checks code quality and runs tests.

**What it does:**

- Runs flake8 for linting
- Checks code formatting with black and isort
- Validates feature store schema and parity
- Runs all our unit tests on Python 3.11 and 3.12
- Runs SonarCloud static code analysis

If any of these fail, the pipeline stops and we get notified. This prevents bad code from going further.

**SonarCloud Integration:**

We added SonarCloud to automatically scan for bugs, vulnerabilities, and code smells. Results are visible at:
https://sonarcloud.io/project/overview?id=sudheer628_Group75-MLops-Assignment

---

### 2. Container Build (container-build.yml)

This workflow only runs after CI passes. It builds our Docker image and pushes it to GitHub Container Registry.

**What it does:**

- Builds the Docker image from our Dockerfile
- Tests that the container starts properly
- Pushes to GHCR with tags like `latest`, `main`, and the commit SHA

Our container is stored at: `ghcr.io/sudheer628/group75-mlops-assignment/heart-disease-api`

---

### 3. Deploy to GCP (deploy.yml)

This is the final step in our deployment chain. It automatically deploys to our GCP VM after the container is built.

**What it does:**

- SSHs into our GCP VM
- Pulls the latest code from GitHub
- Pulls the new Docker image
- Restarts the containers using docker-compose
- Runs a health check to make sure everything is working

**GitHub Secrets we needed to set up:**

- `GCP_VM_HOST` - Our VM's IP address
- `GCP_VM_USER` - SSH username
- `GCP_VM_SSH_KEY` - Private SSH key for authentication

---

### 4. PR Validation (pr-validation.yml)

This runs when someone opens a pull request. It's a lighter version of CI that gives quick feedback.

**What it does:**

- Quick lint check for syntax errors
- Runs Task 1 and Task 2 tests (the critical ones)
- Security scan

We made this separate from the main CI so PR authors get faster feedback without waiting for the full test suite.

---

### 5. Model Training (model-training.yml)

This workflow is different from the others - it doesn't run automatically on every push.

**When it runs:**

- Manually when we click "Run workflow" in GitHub Actions
- Automatically every Monday at 2 AM UTC

**What it does:**

- Runs the complete ML pipeline (data acquisition, feature engineering, training)
- Uploads models to MLflow on Railway
- Packages models for deployment

We kept this separate because model training is slow and expensive. We don't want to retrain on every code change.

---

## How the Workflows Chain Together

The key thing we learned is using `workflow_run` to chain workflows. Here's how it works:

```yaml
# In container-build.yml
on:
  workflow_run:
    workflows: ["CI Pipeline - Lint and Test"]
    types: [completed]

# In deploy.yml
on:
  workflow_run:
    workflows: ["Build and Push Container"]
    types: [completed]
```

This means:

- Container build waits for CI to finish
- Deploy waits for container build to finish
- If any step fails, the chain stops

---

## Manual Triggers

All our workflows can also be triggered manually from the GitHub Actions UI. This is useful when:

- We want to deploy without making code changes
- We need to retrain models on demand
- A workflow failed and we want to retry

To do this: Go to Actions tab -> Select workflow -> Click "Run workflow"

---

## Container Registry

Our Docker images are stored in GitHub Container Registry (GHCR). After each successful build, you can find the image at:

```
ghcr.io/sudheer628/group75-mlops-assignment/heart-disease-api:latest
```

To pull and run it locally:

```bash
docker pull ghcr.io/sudheer628/group75-mlops-assignment/heart-disease-api:latest
docker run -p 8000:8000 ghcr.io/sudheer628/group75-mlops-assignment/heart-disease-api:latest
```

---

## What We Learned

Setting up this CI/CD pipeline taught us a few things:

1. **Workflow chaining is important** - We initially had workflows running in parallel and conflicting with each other. Using `workflow_run` fixed this.

2. **Keep workflows focused** - Each workflow does one thing well. CI does testing, container-build does building, deploy does deploying.

3. **Separate model training from deployment** - ML training is slow and shouldn't block regular code deployments.

4. **Health checks are essential** - The deploy workflow checks if the API is healthy after deployment. This catches issues early.

5. **Secrets management** - GitHub Secrets is a secure way to store credentials like SSH keys and API tokens.

---

## Summary

Our CI/CD pipeline automates the entire process from code push to production deployment:

```
git push origin main
    |
    +-> CI Pipeline (~2-3 min) - Tests pass?
    |       |
    |       +-> SonarCloud Analysis (parallel)
    |       |
    +-> Container Build (~3-5 min) - Image built?
    |       |
    +-> Deploy to GCP (~1-2 min) - Health check passes?
            |
            v
    API live at myprojectdemo.online
```

Total time: About 6-10 minutes from push to production.

Code quality results available at: https://sonarcloud.io/project/overview?id=sudheer628_Group75-MLops-Assignment
