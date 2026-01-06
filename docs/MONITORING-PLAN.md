# TASK-8: Monitoring & Logging with Grafana Cloud

## Status: ✅ COMPLETED

## Overview

**Objective**: Implement monitoring and logging for the Heart Disease Prediction API using Grafana Cloud

**Tools**:

- **Grafana Cloud**: Hosted monitoring platform (https://group75mlops.grafana.net/)
- **Grafana Alloy**: Lightweight agent for metrics & logs collection
- **FastAPI**: Built-in metrics exposure via `/metrics` endpoint

**Architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│                         GCP VM                                  │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │ heart-disease-  │    │   nginx-proxy   │                    │
│  │      api        │    │   (container)   │                    │
│  │  (container)    │    │                 │                    │
│  │  Port 8000      │    │   Port 80/443   │                    │
│  │  /metrics       │    │                 │                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           │                      │                              │
│           └──────────┬───────────┘                              │
│                      │ Docker logs                              │
│                      ▼                                          │
│           ┌─────────────────────┐                              │
│           │   Grafana Alloy     │                              │
│           │   (systemd service) │                              │
│           │                     │                              │
│           │ • Scrapes /metrics  │                              │
│           │ • Collects Docker   │                              │
│           │   container logs    │                              │
│           └──────────┬──────────┘                              │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼ HTTPS
         ┌─────────────────────────────┐
         │      Grafana Cloud          │
         │  group75mlops.grafana.net   │
         │                             │
         │  ┌─────────┐ ┌───────────┐  │
         │  │Prometheus│ │   Loki    │  │
         │  │ Metrics  │ │   Logs    │  │
         │  └────┬─────┘ └─────┬─────┘  │
         │       │             │        │
         │       ▼             ▼        │
         │  ┌─────────────────────┐     │
         │  │    Dashboards &     │     │
         │  │       Alerts        │     │
         │  └─────────────────────┘     │
         └─────────────────────────────┘
```

---

## TASK-8 Requirements

| Requirement       | Implementation                    | Status      |
| ----------------- | --------------------------------- | ----------- |
| Monitoring        | Prometheus metrics via Alloy      | ✅ Complete |
| Logging           | Docker container logs via Alloy   | ✅ Complete |
| Visualization     | Grafana Cloud dashboards          | ✅ Complete |
| Alerts            | Threshold-based alerts (optional) | ✅ Optional |
| Cloud Integration | Grafana Cloud                     | ✅ Complete |

---

## Implementation Summary

### Grafana Cloud Configuration

- **Organization**: `group75mlops`
- **URL**: https://group75mlops.grafana.net/
- **Metrics Endpoint**: `https://prometheus-prod-43-prod-ap-south-1.grafana.net/api/prom/push`
- **Logs Endpoint**: `https://logs-prod-028.grafana.net/loki/api/v1/push`
- **Fleet Management**: `https://fleet-management-prod-018.grafana.net`

### GCP VM Configuration

- **VM Name**: `my-free-vm`
- **Alloy Service**: Running as systemd service
- **Config Location**: `/etc/alloy/config.alloy`

---

## Alloy Configuration (Final Working Version)

**File: `/etc/alloy/config.alloy`** on GCP VM:

```alloy
// ============================================
// Remote Configuration
// ============================================
remotecfg {
  url            = "https://fleet-management-prod-018.grafana.net"
  id             = "my-free-vm"
  poll_frequency = "60s"
  basic_auth {
    username = "1480365"
    password = sys.env("GCLOUD_RW_API_KEY")
  }
}

// ============================================
// Prometheus Metrics
// ============================================
prometheus.remote_write "metrics_service" {
  endpoint {
    url = "https://prometheus-prod-43-prod-ap-south-1.grafana.net/api/prom/push"
    basic_auth {
      username = "2884963"
      password = sys.env("GCLOUD_RW_API_KEY")
    }
  }
}

prometheus.scrape "heart_disease_api" {
  targets         = [{"__address__" = "localhost:8000"}]
  metrics_path    = "/metrics"
  scrape_interval = "15s"
  scrape_timeout  = "10s"
  forward_to      = [prometheus.remote_write.metrics_service.receiver]
}

// ============================================
// Loki Logs
// ============================================
loki.write "grafana_cloud_loki" {
  endpoint {
    url = "https://logs-prod-028.grafana.net/loki/api/v1/push"
    basic_auth {
      username = "1438158"
      password = sys.env("GCLOUD_RW_API_KEY")
    }
  }
}

// Docker container logs collection
discovery.docker "containers" {
  host = "unix:///var/run/docker.sock"
}

loki.source.docker "docker_logs" {
  host       = "unix:///var/run/docker.sock"
  targets    = discovery.docker.containers.targets
  labels     = {
    job  = "docker",
    host = "my-free-vm",
  }
  forward_to = [loki.process.docker_logs.receiver]
}

// Process and relabel docker logs
loki.process "docker_logs" {
  stage.docker {}

  stage.labels {
    values = {
      container_name = "",
    }
  }

  forward_to = [loki.write.grafana_cloud_loki.receiver]
}
```

---

## LogQL Queries for Grafana

### Basic Queries

```logql
# All Docker logs
{job="docker"}

# Filter by log content
{job="docker"} |= "predict"
{job="docker"} |= "health"
{job="docker"} |= "uvicorn"
```

### Error Filtering

```logql
# Errors (case-insensitive)
{job="docker"} |~ "(?i)error"

# Multiple log levels
{job="docker"} |~ "ERROR|WARN|CRITICAL"

# HTTP errors (4xx, 5xx)
{job="docker"} |~ "4[0-9]{2}|5[0-9]{2}"
```

### Exclude Noise

```logql
# Exclude health checks
{job="docker"} != "health"

# Exclude successful responses
{job="docker"} != "200"
```

### Rate/Count Queries

```logql
# Error rate over time
count_over_time({job="docker"} |= "error" [5m])

# Request rate
rate({job="docker"} [1m])
```

---

## Prometheus Queries for Dashboards

### Prediction Metrics

```promql
# Total predictions rate
sum(rate(heart_disease_predictions_total[5m]))

# Predictions by result (healthy vs disease)
sum by (prediction_result) (rate(heart_disease_predictions_total[5m]))

# Prediction latency (95th percentile)
histogram_quantile(0.95, rate(heart_disease_prediction_latency_seconds_bucket[5m]))
```

### API Performance

```promql
# API request rate
sum(rate(api_requests_total[5m]))

# Error rate
sum(rate(api_errors_total[5m]))

# Model loaded status
heart_disease_model_loaded
```

---

## Dashboards Created

### 1. Heart Disease API Monitoring Dashboard

**Panels:**

- Total Predictions (Stat)
- Predictions by Result - Healthy vs Disease (Pie Chart)
- Prediction Latency 95th Percentile (Graph)
- API Requests by Status (Graph)
- Model Status (Gauge)
- Error Rate (Graph)

### 2. Logs Dashboard

**Panels:**

- Live Docker Logs Stream
- Error Logs Filter
- Request Logs

---

## Alloy Management Commands

```bash
# Check Alloy status
sudo systemctl status alloy

# View Alloy logs
sudo journalctl -u alloy -f

# Restart Alloy
sudo systemctl restart alloy

# Reload configuration
sudo systemctl reload alloy
```

---

## Troubleshooting

### Issue: "timestamp too old" errors

**Cause**: Grafana Cloud Loki rejects logs older than ~7 days

**Solution**:

```bash
# Clear old Docker logs
sudo sh -c 'truncate -s 0 /var/lib/docker/containers/*/*-json.log'

# Clear Alloy position files
sudo rm -rf /var/lib/alloy/data/loki.source.docker.*

# Restart Alloy
sudo systemctl restart alloy
```

### Issue: Docker logs not appearing

**Solution**:

```bash
# Ensure Alloy has Docker socket access
sudo usermod -aG docker alloy
sudo systemctl restart alloy
```

### Issue: Metrics not appearing

**Solution**:

```bash
# Verify API metrics endpoint
curl http://localhost:8000/metrics

# Check Alloy scrape status
sudo journalctl -u alloy | grep -i scrape
```

---

## What's Monitored

### Metrics Collected

| Metric                                     | Description                     |
| ------------------------------------------ | ------------------------------- |
| `heart_disease_predictions_total`          | Total predictions by result     |
| `heart_disease_prediction_latency_seconds` | Prediction latency histogram    |
| `api_requests_total`                       | API requests by endpoint/status |
| `api_request_latency_seconds`              | API request latency             |
| `heart_disease_model_loaded`               | Model loaded status (0/1)       |
| `api_errors_total`                         | Total API errors by type        |

### Logs Collected

| Source            | Label        | Description             |
| ----------------- | ------------ | ----------------------- |
| heart-disease-api | job="docker" | API application logs    |
| nginx-proxy       | job="docker" | Nginx access/error logs |

---

## Files Modified for Monitoring

| File                   | Changes                               |
| ---------------------- | ------------------------------------- |
| `app/metrics.py`       | Prometheus metrics definitions        |
| `app/main.py`          | `/metrics` endpoint, metric recording |
| `requirements-api.txt` | Added `prometheus-client>=0.19.0`     |

---

## Summary

| Phase                          | Duration | Status      |
| ------------------------------ | -------- | ----------- |
| Phase 1: Grafana Cloud Setup   | 10 min   | ✅ Complete |
| Phase 2: Install Grafana Alloy | 5 min    | ✅ Complete |
| Phase 3: FastAPI Metrics       | 20 min   | ✅ Complete |
| Phase 4: Configure Alloy       | 10 min   | ✅ Complete |
| Phase 5: Dashboards            | 15 min   | ✅ Complete |
| Phase 6: Alerts                | 10 min   | ✅ Optional |
| **Total**                      | ~70 min  | ✅ Complete |

---

## Access URLs

- **Grafana Cloud**: https://group75mlops.grafana.net/
- **API Health**: http://myprojectdemo.online/health
- **API Metrics**: http://myprojectdemo.online/metrics (internal: localhost:8000/metrics)
- **API Docs**: http://myprojectdemo.online/docs

**TASK-8: Monitoring & Logging - COMPLETED ✅**
