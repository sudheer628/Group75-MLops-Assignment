# TASK-8: Monitoring & Logging with Grafana Cloud

## Overview

**Objective**: Implement monitoring and logging for the Heart Disease Prediction API using Grafana Cloud

**Tools**:

- **Grafana Cloud**: Hosted monitoring platform (https://group75mlops.grafana.net/)
- **Grafana Alloy**: Lightweight agent for metrics & logs collection
- **FastAPI**: Built-in metrics exposure

**Architecture**:

```
FastAPI Application
    ↓
Prometheus Metrics (Port 9090)
    ↓
Grafana Alloy (Collects metrics & logs)
    ↓
Grafana Cloud (Visualization & Alerts)
    ↓
Dashboards & Alerts
```

---

## TASK-8 Requirements

| Requirement       | Implementation               | Status |
| ----------------- | ---------------------------- | ------ |
| Monitoring        | Prometheus metrics via Alloy | ✓      |
| Logging           | API request/response logs    | ✓      |
| Visualization     | Grafana Cloud dashboards     | ✓      |
| Alerts            | Threshold-based alerts       | ✓      |
| Cloud Integration | Grafana Cloud                | ✓      |

---

## Phase 1: Grafana Cloud Setup (10 minutes)

### Step 1.1: Access Grafana Cloud

1. Go to: https://group75mlops.grafana.net/
2. Login with your credentials
3. Navigate to: **Connections → Configure**

### Step 1.2: Generate API Token

**In Grafana Cloud UI:**

1. Click **Account** (bottom left) → **API tokens**
2. Click **New API token**
3. Name: `alloy-metrics-logs`
4. Role: `MetricsPublisher`
5. Click **Create token**
6. **Copy and save** the token (format: `glc_xxxxxxxxxxxxxxxxxxxx`)

### Step 1.3: Get Alloy Installation Command

**In Grafana Cloud UI:**

1. Go to **Connections → Configure**
2. Look for **Grafana Alloy** section
3. You'll see a complete installation command with:
   - `GCLOUD_HOSTED_METRICS_URL` - Metrics endpoint
   - `GCLOUD_HOSTED_LOGS_URL` - Logs endpoint
   - `GCLOUD_RW_API_KEY` - Your API token (already included!)

**Example:**

```bash
GCLOUD_HOSTED_METRICS_ID="2884963" \
GCLOUD_HOSTED_METRICS_URL="https://prometheus-prod-43-prod-ap-south-1.grafana.net/api/prom/push" \
GCLOUD_HOSTED_LOGS_ID="1438158" \
GCLOUD_HOSTED_LOGS_URL="https://logs-prod-028.grafana.net/loki/api/v1/push" \
GCLOUD_FM_URL="https://fleet-management-prod-018.grafana.net" \
GCLOUD_FM_POLL_FREQUENCY="60s" \
GCLOUD_FM_HOSTED_ID="1480365" \
ARCH="amd64" \
GCLOUD_RW_API_KEY="glc_xxxxxxxxxxxxxxxxxxxx" \
/bin/sh -c "$(curl -fsSL https://storage.googleapis.com/cloud-onboarding/alloy/scripts/install-linux.sh)"
```

---

## Phase 2: Install Grafana Alloy on GCP VM (5 minutes)

### Step 2.1: SSH into GCP VM

```bash
gcloud compute ssh INSTANCE_NAME --zone=us-central1-a
```

### Step 2.2: Run Alloy Installation

Copy the complete command from Phase 1, Step 1.3 and run it on GCP VM:

```bash
GCLOUD_HOSTED_METRICS_ID="2884963" \
GCLOUD_HOSTED_METRICS_URL="https://prometheus-prod-43-prod-ap-south-1.grafana.net/api/prom/push" \
GCLOUD_HOSTED_LOGS_ID="1438158" \
GCLOUD_HOSTED_LOGS_URL="https://logs-prod-028.grafana.net/loki/api/v1/push" \
GCLOUD_FM_URL="https://fleet-management-prod-018.grafana.net" \
GCLOUD_FM_POLL_FREQUENCY="60s" \
GCLOUD_FM_HOSTED_ID="1480365" \
ARCH="amd64" \
GCLOUD_RW_API_KEY="glc_xxxxxxxxxxxxxxxxxxxx" \
/bin/sh -c "$(curl -fsSL https://storage.googleapis.com/cloud-onboarding/alloy/scripts/install-linux.sh)"
```

**What this does:**

- Downloads Grafana Alloy
- Installs as systemd service
- Configures metrics endpoint
- Configures logs endpoint
- Sets up API authentication

### Step 2.3: Verify Installation

```bash
# Check if Alloy is running
sudo systemctl status alloy

# View logs
sudo journalctl -u alloy -f

# Expected output:
# Active: active (running)
```

### Step 2.4: Test Connection in Grafana Cloud

**In Grafana Cloud UI:**

1. Go to **Connections → Configure**
2. Click **Test Alloy connection** button
3. Should show: "Connection successful"

---

## Phase 3: Code Changes - Add Metrics to FastAPI

### Step 3.1: Code Changes Completed

The following code changes have been implemented:

**1. Created `app/metrics.py`**

Defines all Prometheus metrics:

- `heart_disease_predictions_total` - Total predictions by result (healthy/disease)
- `heart_disease_prediction_latency_seconds` - Prediction latency histogram
- `api_requests_total` - Total API requests by endpoint/method/status
- `api_request_latency_seconds` - API request latency histogram
- `heart_disease_model_loaded` - Model loaded status (gauge)
- `heart_disease_model_inference_seconds` - Model inference time
- `api_errors_total` - Total API errors by type
- `heart_disease_prediction_confidence` - Prediction confidence scores

Helper functions:

- `record_prediction()` - Record prediction metrics
- `record_api_request()` - Record API request metrics
- `record_api_error()` - Record error metrics
- `set_model_loaded()` - Set model status
- `record_model_inference_time()` - Record inference time

**2. Updated `app/main.py`**

- Added imports for metrics module
- Added `/metrics` endpoint (Prometheus scrape endpoint)
- Updated `/predict` endpoint to record metrics:
  - Prediction result and confidence
  - Request latency
  - Error tracking
  - HTTP status codes

**3. Updated `requirements-api.txt`**

- Added `prometheus-client>=0.19.0` dependency

### Step 3.2: Push Code Changes to GitHub

```bash
# From your local machine
git add app/metrics.py
git add app/main.py
git add requirements-api.txt
git commit -m "TASK-8: Add Prometheus metrics for Grafana Cloud monitoring"
git push origin main
```

**Expected**: GitHub Actions CI/CD pipeline runs and passes all checks

---

## Phase 3.3: Wait for CI/CD Pipeline

**What happens**:

1. Code linting (black, isort, flake8)
2. Tests run
3. Docker image builds
4. Image pushed to GHCR

**Expected**: All checks pass ✓

---

## Phase 4: Configure Alloy to Scrape Metrics (10 minutes)

### Step 4.1: Create Alloy Configuration

**File: `/etc/alloy/config.alloy`** (on GCP VM)

```alloy
prometheus.scrape "heart_disease_api" {
  targets = [{"__address__" = "localhost:8000"}]
  metrics_path = "/metrics"
  scrape_interval = "15s"
  scrape_timeout = "10s"

  forward_to = [prometheus.remote_write.grafana_cloud.receiver]
}

prometheus.remote_write "grafana_cloud" {
  endpoint {
    url = env("GCLOUD_HOSTED_METRICS_URL")

    headers = {
      "Authorization" = "Bearer " + env("GCLOUD_RW_API_KEY"),
    }
  }
}

loki.source.syslog "api_logs" {
  listen_address = "127.0.0.1:514"

  forward_to = [loki.write.grafana_cloud.receiver]
}

loki.write "grafana_cloud" {
  endpoint {
    url = env("GCLOUD_HOSTED_LOGS_URL")

    headers = {
      "Authorization" = "Bearer " + env("GCLOUD_RW_API_KEY"),
    }
  }
}
```

### Step 4.2: Reload Alloy Configuration

```bash
# Reload Alloy with new configuration
sudo systemctl reload alloy

# Verify it's still running
sudo systemctl status alloy
```

---

## Phase 5: Create Grafana Dashboards (15 minutes)

### Step 5.1: Create Dashboard

**In Grafana Cloud UI:**

1. Click **Dashboards** (left sidebar)
2. Click **New → New Dashboard**
3. Click **Add visualization**
4. Select **Prometheus** as data source

### Step 5.2: Add Panels

**Panel 1: Total Predictions**

```
Query: sum(rate(heart_disease_predictions_total[5m]))
Visualization: Stat
Title: Predictions per 5 minutes
```

**Panel 2: Prediction Latency**

```
Query: histogram_quantile(0.95, rate(heart_disease_prediction_latency_seconds_bucket[5m]))
Visualization: Graph
Title: 95th Percentile Latency
```

**Panel 3: Predictions by Result**

```
Query: sum by (prediction_result) (rate(heart_disease_predictions_total[5m]))
Visualization: Pie Chart
Title: Healthy vs Disease Predictions
```

**Panel 4: API Requests**

```
Query: sum by (status) (rate(api_requests_total[5m]))
Visualization: Graph
Title: API Requests by Status
```

### Step 5.3: Save Dashboard

1. Click **Save** (top right)
2. Name: `Heart Disease API Monitoring`
3. Click **Save**

---

## Phase 6: Setup Alerts (Optional)

### Step 6.1: Create Alert Rule

**In Grafana Cloud UI:**

1. Go to **Alerting → Alert Rules**
2. Click **New alert rule**
3. Configure:
   - **Name**: `High Prediction Latency`
   - **Query**: `histogram_quantile(0.95, rate(heart_disease_prediction_latency_seconds_bucket[5m])) > 1`
   - **Condition**: `> 1 second`
   - **For**: `5 minutes`
4. Click **Save**

### Step 6.2: Setup Notification Channel

1. Go to **Alerting → Contact points**
2. Click **New contact point**
3. Select **Email** or **Slack**
4. Configure and save

---

---

## Testing Metrics

### Test 1: Check Metrics Endpoint

```bash
# From local machine or GCP VM
curl http://localhost:8000/metrics

# Expected output: Prometheus metrics in text format
# HELP heart_disease_predictions_total Total number of predictions made
# TYPE heart_disease_predictions_total counter
# heart_disease_predictions_total{prediction_result="healthy"} 0.0
# ...
```

### Test 2: Make Predictions and Check Metrics

```bash
# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3}'

# Check metrics again
curl http://localhost:8000/metrics | grep heart_disease_predictions_total

# Expected: Counter incremented
# heart_disease_predictions_total{prediction_result="disease"} 1.0
```

### Test 3: Check Alloy Status

```bash
# On GCP VM
sudo systemctl status alloy

# View logs
sudo journalctl -u alloy -f

# Expected: Alloy running and scraping metrics
```

---

## Environment Variables on GCP VM

### Option: Export API Token as Environment Variable

Instead of hardcoding in config files, you can export as environment variable:

```bash
# On GCP VM
export GCLOUD_RW_API_KEY="glc_xxxxxxxxxxxxxxxxxxxx"
export GCLOUD_HOSTED_METRICS_URL="https://prometheus-prod-43-prod-ap-south-1.grafana.net/api/prom/push"
export GCLOUD_HOSTED_LOGS_URL="https://logs-prod-028.grafana.net/loki/api/v1/push"

# Verify
echo $GCLOUD_RW_API_KEY
```

**To make persistent:**

```bash
# Add to ~/.bashrc or ~/.profile
echo 'export GCLOUD_RW_API_KEY="glc_xxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
echo 'export GCLOUD_HOSTED_METRICS_URL="https://prometheus-prod-43-prod-ap-south-1.grafana.net/api/prom/push"' >> ~/.bashrc
echo 'export GCLOUD_HOSTED_LOGS_URL="https://logs-prod-028.grafana.net/loki/api/v1/push"' >> ~/.bashrc

# Reload
source ~/.bashrc
```

---

## Troubleshooting

### Issue: Metrics endpoint returns 404

**Solution**:

```bash
# Verify API is running
docker-compose ps

# Check logs
docker logs heart-disease-api

# Verify endpoint
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### Issue: Alloy not collecting metrics

**Solution**:

```bash
# Check Alloy status
sudo systemctl status alloy

# View Alloy logs
sudo journalctl -u alloy -f

# Verify Alloy can reach API
curl http://localhost:8000/metrics
```

### Issue: Metrics not appearing in Grafana Cloud

**Solution**:

1. Verify Alloy is running: `sudo systemctl status alloy`
2. Check Alloy logs: `sudo journalctl -u alloy -f`
3. Verify API token is correct
4. Wait 1-2 minutes for metrics to appear
5. In Grafana Cloud, click **Test Alloy connection**

---

## What's Monitored

### Metrics Collected

**Predictions**:

- Total predictions made
- Predictions by result (healthy vs disease)
- Prediction latency (95th percentile, etc.)
- Prediction confidence scores

**API Performance**:

- Total API requests
- Request latency by endpoint
- HTTP status codes (200, 500, 503, etc.)
- Error rates and types

**Model Health**:

- Model loaded status
- Model inference time
- Error tracking

### Logs Collected (via Grafana Alloy)

**Application Logs**:

- API startup/shutdown
- Prediction requests
- Errors and exceptions
- Model loading status

**System Logs**:

- Alloy agent status
- Metrics collection status
- Connection health

### What's NOT Monitored

✗ TASK-1 logs (data acquisition) - one-time, not production
✗ TASK-2 logs (model training) - one-time, not production
✗ TASK-3 logs (experiment tracking) - handled by MLflow
✗ TASK-4 logs (model packaging) - one-time, not production
✗ CI/CD logs - handled by GitHub Actions

---

## Prometheus Queries for Grafana Dashboards

### Total Predictions

```
sum(rate(heart_disease_predictions_total[5m]))
```

### Predictions by Result

```
sum by (prediction_result) (rate(heart_disease_predictions_total[5m]))
```

### Prediction Latency (95th percentile)

```
histogram_quantile(0.95, rate(heart_disease_prediction_latency_seconds_bucket[5m]))
```

### API Request Rate

```
sum(rate(api_requests_total[5m]))
```

### Error Rate

```
sum(rate(api_errors_total[5m]))
```

### Model Status

```
heart_disease_model_loaded
```

---

## Approach Comparison

### **Grafana Alloy (Recommended - What We're Using)**

**Pros:**

- One-liner installation
- Automatic configuration
- Includes metrics + logs
- Grafana manages everything
- Simpler setup

**Cons:**

- Less control
- Newer tool

---

## Information Required from You

Before proceeding with GCP VM deployment, provide:

1. **Grafana Cloud Organization Name**:

   - Answer: `group75mlops`

2. **Metrics URL** (from Alloy command):

   - Example: `https://prometheus-prod-43-prod-ap-south-1.grafana.net/api/prom/push`

3. **Logs URL** (from Alloy command):

   - Example: `https://logs-prod-028.grafana.net/loki/api/v1/push`

4. **API Token**:

   - Format: `glc_xxxxxxxxxxxxxxxxxxxx`

5. **Deployment Environment**:
   - [ ] GCP VM only
   - [ ] Local + GCP VM

---

## Approach Comparison

### **Grafana Alloy (Recommended - What We're Using)**

**Pros:**

- One-liner installation
- Automatic configuration
- Includes metrics + logs
- Grafana manages everything
- Simpler setup

**Cons:**

- Less control
- Newer tool

---

## Information Required from You

Before proceeding with code changes, please provide:

1. **Grafana Cloud Organization Name**:

   - Answer: `group75mlops`

2. **Metrics URL** (from Alloy command):

   - Example: `https://prometheus-prod-43-prod-ap-south-1.grafana.net/api/prom/push`

3. **Logs URL** (from Alloy command):

   - Example: `https://logs-prod-028.grafana.net/loki/api/v1/push`

4. **API Token**:

   - Format: `glc_xxxxxxxxxxxxxxxxxxxx`

5. **Deployment Environment**:
   - [ ] GCP VM only
   - [ ] Local + GCP VM

---

## Next Steps

1. **Complete Grafana Cloud Setup** (Phase 1)
2. **Install Grafana Alloy** (Phase 2)
3. **Test Alloy Connection** (Phase 2, Step 2.4)
4. **Provide Information** (above section)
5. **Confirm Approach**
6. **Proceed with Code Changes**

---

## Summary

| Phase                          | Duration    | Status      |
| ------------------------------ | ----------- | ----------- |
| Phase 1: Grafana Cloud Setup   | 10 min      | Pending     |
| Phase 2: Install Grafana Alloy | 5 min       | Pending     |
| Phase 3: FastAPI Metrics       | 20 min      | Pending     |
| Phase 4: Configure Alloy       | 10 min      | Pending     |
| Phase 5: Dashboards            | 15 min      | Pending     |
| Phase 6: Alerts                | 10 min      | Optional    |
| **Total**                      | **~70 min** | **Pending** |

**Ready to proceed after your confirmation!**
