# TASK-7 Production Deployment Plan

## GCP VM Deployment with Ephemeral IP & Domain Routing

---

## Deployment Overview

**Target Environment:**

- GCP VM with Ephemeral IP: `35.233.155.69`
- Domain: `myprojectdemo.online`
- Container Registry: GHCR (`ghcr.io/sudheer628/group75-mlops-assignment/heart-disease-api`)
- Deployment Method: Docker Compose
- Reverse Proxy: Nginx

**Architecture:**

```
User Request
    ↓
Domain (myprojectdemo.online)
    ↓
DNS A Record → 35.233.155.69
    ↓
Nginx Reverse Proxy (Port 80/443)
    ↓
Docker Container (Port 8000)
    ↓
FastAPI Application
    ↓
Response
```

---

## Assignment Requirements Met

| Requirement              | Implementation      | Status |
| ------------------------ | ------------------- | ------ |
| Deploy to public cloud   | GCP VM              | ✓ YES  |
| Use deployment manifest  | Docker Compose YAML | ✓ YES  |
| Expose via Load Balancer | Nginx reverse proxy | ✓ YES  |
| Verify endpoints         | Testing scenarios   | ✓ YES  |
| Provide screenshots      | Documentation       | ✓ YES  |

---

## Phase 1: GCP VM Setup (30 minutes)

### Step 1.1: SSH into GCP VM

```bash
gcloud compute ssh INSTANCE_NAME --zone=us-central1-a
```

### Step 1.2: Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Step 1.3: Install Docker

```bash
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
newgrp docker
```

### Step 1.4: Install Docker Compose

```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

### Step 1.5: Stop System Nginx (Important!)

The system nginx installed in Step 1.5 is **NOT used** in this deployment. We use **Docker nginx** instead.

Stop the system nginx to avoid port conflicts:

```bash
# Stop the system nginx service
sudo systemctl stop nginx

# Disable it from auto-starting
sudo systemctl disable nginx

# Verify it's stopped
sudo systemctl status nginx
```

**Why?** Docker nginx container needs port 80, and system nginx would block it.

### Step 1.6: Install Git

```bash
sudo apt-get install -y git
```

---

## Phase 2: Clone Repository & Setup (5 minutes)

### Step 2.1: Clone Repository

```bash
cd /home/$USER
git clone https://github.com/sudheer628/group75-mlops-assignment.git
cd group75-mlops-assignment
```

**Files already included in repository:**

- `docker-compose.yml` - Production-ready Docker Compose configuration
- `nginx.conf` - Nginx reverse proxy configuration (for Docker container)

### Step 2.2: Understand Nginx Setup

**Important:** This deployment uses **Docker Nginx**, not system nginx.

- `nginx.conf` is mounted into Docker container at `/etc/nginx/conf.d/default.conf`
- Docker automatically loads this configuration when container starts
- **No manual nginx config directory setup needed**

### Step 2.3: Verify Docker Compose Configuration

The repository already contains `docker-compose.yml` with:

- API service pulling pre-built image from GHCR
- API exposed on localhost:8000 (internal only)
- **Nginx service (Docker container)** exposed on 0.0.0.0:80/443 (public)
- Volume mount: `./nginx.conf:/etc/nginx/conf.d/default.conf:ro`
- Custom network for container communication
- Health checks and auto-restart policies

### Step 2.4: Verify Nginx Configuration File

The repository already contains `nginx.conf` with:

- Reverse proxy routing to API container (`proxy_pass http://api:8000`)
- Domain configuration for `myprojectdemo.online` and `www.myprojectdemo.online`
- Proper proxy headers for client IP tracking
- Health check endpoint passthrough

### Step 2.5: Configure GCP Firewall

```bash
# Allow HTTP (port 80)
gcloud compute firewall-rules create allow-http \
  --allow=tcp:80 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=http-server

# Allow HTTPS (port 443)
gcloud compute firewall-rules create allow-https \
  --allow=tcp:443 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=https-server

# Apply tags to VM
gcloud compute instances add-tags INSTANCE_NAME \
  --tags=http-server,https-server \
  --zone=us-central1-a
```

---

## Phase 3: Configure Domain DNS (5 minutes)

### Step 3.1: Add DNS A Record

**In your domain registrar (myprojectdemo.online):**

| Type | Name | Value         | TTL  |
| ---- | ---- | ------------- | ---- |
| A    | @    | 35.233.155.69 | 3600 |
| A    | www  | 35.233.155.69 | 3600 |

### Step 3.2: Verify DNS Resolution

```bash
nslookup myprojectdemo.online
dig myprojectdemo.online
```

**Expected output:**

```
myprojectdemo.online has address 35.233.155.69
```

---

## Phase 4: Deploy Docker Containers (10 minutes)

### Step 4.1: Pull Latest Image

```bash
docker pull ghcr.io/sudheer628/group75-mlops-assignment/heart-disease-api:latest
```

### Step 4.2: Verify System Nginx is Stopped

**Critical:** Make sure system nginx is not running (it blocks port 80):

```bash
# Check if system nginx is running
sudo systemctl status nginx

# If it's running, stop it
sudo systemctl stop nginx
sudo systemctl disable nginx
```

### Step 4.3: Verify Nginx Configuration File

The `nginx.conf` file in the repository will be automatically mounted into the Docker container:

```bash
# Verify nginx.conf exists in repository
cat nginx.conf

# Expected output should show:
# - server block listening on port 80
# - proxy_pass http://api:8000 (routes to API container)
# - server_name myprojectdemo.online www.myprojectdemo.online
```

### Step 4.4: Start Docker Compose

```bash
docker-compose up -d
```

**What happens automatically:**

1. Pulls API image from GHCR
2. Starts API container on internal port 8000
3. Starts Nginx container (Docker image)
4. **Automatically mounts** `./nginx.conf` into container at `/etc/nginx/conf.d/default.conf`
5. Nginx reads the configuration and starts routing traffic
6. Creates custom network for container communication

**No manual nginx config directory setup needed** - Docker handles it!

### Step 4.5: Verify Containers Running

```bash
docker-compose ps
docker logs heart-disease-api
docker logs nginx-proxy
```

**Expected output:**

```
NAME                COMMAND                  SERVICE             STATUS
heart-disease-api   "python -m uvicorn..."   api                 Up 2 minutes (healthy)
nginx-proxy         "/docker-entrypoint..."  nginx               Up 2 minutes
```

### Step 4.6: Verify Nginx Configuration is Loaded

```bash
# Check Nginx configuration syntax inside container
docker exec nginx-proxy nginx -t

# Expected output:
# nginx: the configuration file /etc/nginx/conf.d/default.conf syntax is ok
# nginx: configuration file /etc/nginx/conf.d/default.conf test is successful
```

### Step 4.7: Reload Nginx Configuration (if needed)

If you modify `nginx.conf` after containers are running:

```bash
# Reload Nginx without restarting container
docker exec nginx-proxy nginx -s reload

# Or restart the entire docker-compose stack
docker-compose restart nginx
```

---

## How Nginx Configuration Works

**Flow:**

```
Repository: ./nginx.conf
    ↓
docker-compose.yml volume mount
    ↓
Docker container: /etc/nginx/conf.d/default.conf
    ↓
Nginx reads configuration
    ↓
Nginx listens on port 80/443
    ↓
Routes traffic to http://api:8000
```

**Key Points:**

- `nginx.conf` is in your repository root
- Docker automatically mounts it when container starts
- No manual file copying needed
- No manual nginx reload needed (happens automatically on container start)
- If you modify `nginx.conf`, use `docker exec nginx-proxy nginx -s reload`

---

## Phase 5: Auto-Pull Latest Changes from GitHub Actions

### Step 5.1: Create Update Script

**File: `/home/$USER/update-api.sh`**

```bash
#!/bin/bash

# Script to pull latest changes and restart containers
# Called by GitHub Actions webhook or cron job

set -e

LOG_FILE="/var/log/heart-disease-api-update.log"
REPO_DIR="/home/$USER/group75-mlops-assignment"

echo "[$(date)] Starting API update..." >> $LOG_FILE

cd $REPO_DIR

# Pull latest code
echo "[$(date)] Pulling latest code from GitHub..." >> $LOG_FILE
git pull origin main >> $LOG_FILE 2>&1

# Pull latest Docker image
echo "[$(date)] Pulling latest Docker image..." >> $LOG_FILE
docker pull ghcr.io/sudheer628/group75-mlops-assignment/heart-disease-api:latest >> $LOG_FILE 2>&1

# Restart containers
echo "[$(date)] Restarting Docker containers..." >> $LOG_FILE
docker-compose down >> $LOG_FILE 2>&1
docker-compose up -d >> $LOG_FILE 2>&1

# Verify health
echo "[$(date)] Verifying API health..." >> $LOG_FILE
sleep 10
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "[$(date)] API is healthy. Update completed successfully." >> $LOG_FILE
else
    echo "[$(date)] ERROR: API health check failed!" >> $LOG_FILE
    exit 1
fi

echo "[$(date)] Update completed." >> $LOG_FILE
```

### Step 5.2: Make Script Executable

```bash
chmod +x /home/$USER/update-api.sh
```

### Step 5.3: Setup Cron Job (Auto-pull every 30 minutes)

```bash
crontab -e
```

**Add this line:**

```
*/30 * * * * /home/$USER/update-api.sh
```

### Step 5.4: Setup GitHub Actions Webhook (Optional - Real-time Updates)

**Create webhook trigger in GitHub Actions:**

File: `.github/workflows/deploy-to-gcp.yml`

```yaml
name: Deploy to GCP VM

on:
  push:
    branches: [main]
  workflow_run:
    workflows: ["Build and Push Container"]
    types: [completed]
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    if: github.event.workflow_run.conclusion == 'success' || github.event_name == 'push'

    steps:
      - name: Deploy to GCP VM
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.GCP_VM_IP }}
          username: ${{ secrets.GCP_VM_USER }}
          key: ${{ secrets.GCP_VM_SSH_KEY }}
          script: |
            cd /home/${{ secrets.GCP_VM_USER }}/group75-mlops-assignment
            /home/${{ secrets.GCP_VM_USER }}/update-api.sh
```

**Add GitHub Secrets:**

- `GCP_VM_IP`: 35.233.155.69
- `GCP_VM_USER`: your-username
- `GCP_VM_SSH_KEY`: your-private-ssh-key

---

## Phase 6: Verify Deployment (10 minutes)

### Step 6.1: Check Container Status

```bash
docker-compose ps
docker-compose logs -f
```

### Step 6.2: Verify Nginx Configuration

```bash
# Check Nginx configuration syntax
docker exec nginx-proxy nginx -t

# View Nginx configuration
docker exec nginx-proxy cat /etc/nginx/conf.d/default.conf

# Check Nginx is listening on ports 80 and 443
docker exec nginx-proxy netstat -tlnp | grep nginx
```

### Step 6.3: Test Local Endpoints (Internal)

```bash
# Health check (internal API)
curl http://localhost:8000/health

# API info (internal API)
curl http://localhost:8000/

# Model info (internal API)
curl http://localhost:8000/model/info
```

### Step 6.4: Test Through Nginx (Localhost)

```bash
# Health check through Nginx
curl http://localhost/health

# API info through Nginx
curl http://localhost/

# Prediction through Nginx
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 3, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
  }'
```

### Step 6.5: Test Through Domain (Public)

Once DNS is propagated (5-30 minutes), test via domain:

```bash
# Health check through domain
curl http://myprojectdemo.online/health

# API info through domain
curl http://myprojectdemo.online/

# Prediction through domain
curl -X POST http://myprojectdemo.online/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 3, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
  }'

# View Nginx access logs
docker logs nginx-proxy
```

---

## Testing Scenarios

### Test 1: CMD (Windows Command Prompt)

```cmd
REM Health check
curl http://myprojectdemo.online/health

REM API info
curl http://myprojectdemo.online/

REM Model info
curl http://myprojectdemo.online/model/info

REM Prediction - High Risk
curl -X POST http://myprojectdemo.online/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"age\": 55, \"sex\": 1, \"cp\": 3, \"trestbps\": 140, \"chol\": 250, \"fbs\": 0, \"restecg\": 1, \"thalach\": 150, \"exang\": 0, \"oldpeak\": 1.5, \"slope\": 2, \"ca\": 0, \"thal\": 3}"

REM Prediction - Low Risk
curl -X POST http://myprojectdemo.online/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"age\": 35, \"sex\": 0, \"cp\": 0, \"trestbps\": 120, \"chol\": 180, \"fbs\": 0, \"restecg\": 0, \"thalach\": 180, \"exang\": 0, \"oldpeak\": 0.0, \"slope\": 1, \"ca\": 0, \"thal\": 2}"
```

### Test 2: PowerShell (Windows)

```powershell
# Health check
Invoke-WebRequest -Uri "http://myprojectdemo.online/health" -Method Get

# API info
Invoke-WebRequest -Uri "http://myprojectdemo.online/" -Method Get

# Model info
Invoke-WebRequest -Uri "http://myprojectdemo.online/model/info" -Method Get

# Prediction - High Risk
$body = @{
    age = 55
    sex = 1
    cp = 3
    trestbps = 140
    chol = 250
    fbs = 0
    restecg = 1
    thalach = 150
    exang = 0
    oldpeak = 1.5
    slope = 2
    ca = 0
    thal = 3
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://myprojectdemo.online/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

# Prediction - Low Risk
$body = @{
    age = 35
    sex = 0
    cp = 0
    trestbps = 120
    chol = 180
    fbs = 0
    restecg = 0
    thalach = 180
    exang = 0
    oldpeak = 0.0
    slope = 1
    ca = 0
    thal = 2
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://myprojectdemo.online/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

### Test 3: Bash/Linux Terminal

```bash
# Health check
curl -v http://myprojectdemo.online/health

# API info
curl -s http://myprojectdemo.online/ | jq .

# Model info
curl -s http://myprojectdemo.online/model/info | jq .

# Prediction - High Risk
curl -X POST http://myprojectdemo.online/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 3, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
  }' | jq .

# Prediction - Low Risk
curl -X POST http://myprojectdemo.online/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "sex": 0, "cp": 0, "trestbps": 120,
    "chol": 180, "fbs": 0, "restecg": 0, "thalach": 180,
    "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 2
  }' | jq .

# Test with verbose output
curl -v -X POST http://myprojectdemo.online/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 3, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
  }'

# Test with timing
curl -w "\nTime: %{time_total}s\n" http://myprojectdemo.online/health
```

### Test 4: Python Requests

```python
import requests
import json

BASE_URL = "http://myprojectdemo.online"

# Health check
response = requests.get(f"{BASE_URL}/health")
print("Health Check:", response.json())

# API info
response = requests.get(f"{BASE_URL}/")
print("API Info:", response.json())

# Model info
response = requests.get(f"{BASE_URL}/model/info")
print("Model Info:", response.json())

# Prediction - High Risk
high_risk_data = {
    "age": 55, "sex": 1, "cp": 3, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
}
response = requests.post(f"{BASE_URL}/predict", json=high_risk_data)
print("High Risk Prediction:", response.json())

# Prediction - Low Risk
low_risk_data = {
    "age": 35, "sex": 0, "cp": 0, "trestbps": 120,
    "chol": 180, "fbs": 0, "restecg": 0, "thalach": 180,
    "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 2
}
response = requests.post(f"{BASE_URL}/predict", json=low_risk_data)
print("Low Risk Prediction:", response.json())
```

### Test 5: Browser Testing

**Visit these URLs in your browser:**

1. **API Documentation (Swagger UI)**

   ```
   http://myprojectdemo.online/docs
   ```

2. **Alternative Documentation (ReDoc)**

   ```
   http://myprojectdemo.online/redoc
   ```

3. **Health Check**

   ```
   http://myprojectdemo.online/health
   ```

4. **API Info**

   ```
   http://myprojectdemo.online/
   ```

5. **Model Info**
   ```
   http://myprojectdemo.online/model/info
   ```

### Test 6: Postman Collection

**Import this collection into Postman:**

```json
{
  "info": {
    "name": "Heart Disease API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "http://myprojectdemo.online/health"
      }
    },
    {
      "name": "API Info",
      "request": {
        "method": "GET",
        "url": "http://myprojectdemo.online/"
      }
    },
    {
      "name": "Model Info",
      "request": {
        "method": "GET",
        "url": "http://myprojectdemo.online/model/info"
      }
    },
    {
      "name": "Predict - High Risk",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "url": "http://myprojectdemo.online/predict",
        "body": {
          "mode": "raw",
          "raw": "{\"age\": 55, \"sex\": 1, \"cp\": 3, \"trestbps\": 140, \"chol\": 250, \"fbs\": 0, \"restecg\": 1, \"thalach\": 150, \"exang\": 0, \"oldpeak\": 1.5, \"slope\": 2, \"ca\": 0, \"thal\": 3}"
        }
      }
    },
    {
      "name": "Predict - Low Risk",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "url": "http://myprojectdemo.online/predict",
        "body": {
          "mode": "raw",
          "raw": "{\"age\": 35, \"sex\": 0, \"cp\": 0, \"trestbps\": 120, \"chol\": 180, \"fbs\": 0, \"restecg\": 0, \"thalach\": 180, \"exang\": 0, \"oldpeak\": 0.0, \"slope\": 1, \"ca\": 0, \"thal\": 2}"
        }
      }
    }
  ]
}
```

---

## Monitoring & Logs

### View Container Logs

```bash
# API logs
docker logs -f heart-disease-api

# Nginx logs
docker logs -f nginx-proxy

# Combined logs
docker-compose logs -f
```

### View Update Script Logs

```bash
sudo tail -f /var/log/heart-disease-api-update.log
```

### Check Container Health

```bash
docker-compose ps
docker inspect heart-disease-api | grep -A 5 "Health"
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs

# Rebuild image
docker-compose down
docker pull ghcr.io/sudheer628/group75-mlops-assignment/heart-disease-api:latest
docker-compose up -d
```

### DNS Not Resolving

```bash
# Verify DNS
nslookup myprojectdemo.online
dig myprojectdemo.online

# Wait for propagation (5-30 minutes)
```

### Nginx Connection Refused

```bash
# Check Nginx container is running
docker-compose ps | grep nginx

# Check Nginx configuration syntax
docker exec nginx-proxy nginx -t

# View Nginx error logs
docker logs nginx-proxy

# Check if Nginx is listening on ports 80/443
docker exec nginx-proxy netstat -tlnp | grep nginx

# Reload Nginx configuration
docker exec nginx-proxy nginx -s reload

# Or restart Nginx container
docker-compose restart nginx
```

### Nginx Not Routing to API

```bash
# Verify nginx.conf is mounted correctly
docker exec nginx-proxy cat /etc/nginx/conf.d/default.conf

# Check if API container is running
docker-compose ps | grep api

# Test API is accessible from Nginx container
docker exec nginx-proxy curl http://api:8000/health

# Check Nginx access logs
docker logs nginx-proxy

# Verify custom network exists
docker network ls | grep api-network

# Inspect network to see connected containers
docker network inspect api-network
```

### API Not Responding

```bash
# Test locally
curl http://localhost:8000/health

# Check firewall
gcloud compute firewall-rules list

# Verify port 80 is open
sudo netstat -tlnp | grep :80

# Check if containers can communicate
docker exec nginx-proxy ping api
```

---

## Expected Test Results

### Health Check Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "ml_model_loaded": true,
  "timestamp": "2025-12-28T10:30:45.123456"
}
```

### High Risk Prediction Response

```json
{
  "prediction": 1,
  "confidence": 0.85,
  "probabilities": [0.15, 0.85],
  "risk_level": "High"
}
```

### Low Risk Prediction Response

```json
{
  "prediction": 0,
  "confidence": 0.92,
  "probabilities": [0.92, 0.08],
  "risk_level": "Low"
}
```

---

## Deployment Complete

**API is now live at:**

- Domain: `http://myprojectdemo.online`
- IP: `35.233.155.69`
- Documentation: `http://myprojectdemo.online/docs`

**Auto-updates enabled:**

- Cron job: Every 30 minutes
- GitHub Actions: On push to main branch

**Ready for TASK-7 submission!**
