# Production Deployment Guide

This document explains how we deployed our Heart Disease Prediction API to a GCP VM and set up automated deployments using GitHub Actions.

---

## Deployment Overview

We deployed our application to a Google Cloud Platform VM with the following setup:

- GCP VM IP: `35.233.155.69`
- Domain: `myprojectdemo.online`
- Container Registry: GitHub Container Registry (GHCR)
- Orchestration: Docker Compose
- Reverse Proxy: Nginx (running in Docker)

Here's how requests flow through our system:

```
User Request
    |
    v
Domain (myprojectdemo.online)
    |
    v
DNS A Record -> 35.233.155.69
    |
    v
Nginx Container (Port 80)
    |
    v
FastAPI Container (Port 8000)
    |
    v
Response
```

---

## Initial GCP VM Setup

These are the one-time setup steps we did on the GCP VM.

### Install Docker and Docker Compose

```bash
# SSH into VM
gcloud compute ssh my-free-vm --zone=<your-zone>

# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Clone the Repository

```bash
cd /home/$USER
git clone https://github.com/sudheer628/group75-mlops-assignment.git
cd group75-mlops-assignment
```

### Configure Firewall

```bash
# Allow HTTP traffic
gcloud compute firewall-rules create allow-http \
  --allow=tcp:80 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=http-server

# Add tag to VM
gcloud compute instances add-tags my-free-vm \
  --tags=http-server \
  --zone=<your-zone>
```

### DNS Configuration

We added an A record in our domain registrar pointing to the VM:

| Type | Name | Value         |
| ---- | ---- | ------------- |
| A    | @    | 35.233.155.69 |
| A    | www  | 35.233.155.69 |

---

## Docker Compose Setup

Our `docker-compose.yml` runs two containers:

1. **heart-disease-api** - Our FastAPI application
2. **nginx-proxy** - Nginx reverse proxy

The nginx container uses our `nginx.conf` file which routes traffic from port 80 to the API on port 8000.

### Start the Application

```bash
docker-compose up -d
```

### Verify Everything is Running

```bash
docker-compose ps
curl http://localhost:8000/health
```

---

## Automated Deployment with GitHub Actions

The cool part is we set up automated deployments. When we push code to main, it automatically deploys to the GCP VM.

### How It Works

We created a `deploy.yml` workflow that:

1. Waits for the container build to complete
2. SSHs into our GCP VM
3. Pulls the latest code and Docker image
4. Restarts the containers
5. Runs a health check

Here's the deployment flow:

```
Push to main
    |
    v
CI Pipeline (tests)
    |
    v
Container Build (builds Docker image)
    |
    v
Deploy to GCP (deploy.yml)
    |
    +-> SSH into VM
    +-> git pull origin main
    +-> docker-compose pull
    +-> docker-compose up -d
    +-> Health check
    |
    v
Live at myprojectdemo.online
```

### GitHub Secrets Required

We added these secrets in GitHub (Settings -> Secrets -> Actions):

| Secret         | Description                        |
| -------------- | ---------------------------------- |
| GCP_VM_HOST    | VM IP address (35.233.155.69)      |
| GCP_VM_USER    | SSH username                       |
| GCP_VM_SSH_KEY | Private SSH key for authentication |

### SSH Key Setup

For GitHub Actions to SSH into our VM, we needed to:

1. Generate an SSH key pair
2. Add the public key to GCP VM metadata (via GCP Console -> Compute Engine -> Metadata -> SSH Keys)
3. Add the private key as a GitHub secret

We used GCP metadata instead of `~/.ssh/authorized_keys` because GCP manages that file and can overwrite it.

---

## Manual Deployment

If we need to deploy manually (without GitHub Actions), we can SSH into the VM and run:

```bash
cd ~/group75-mlops-assignment
git pull origin main
docker-compose pull
docker-compose up -d
```

---

## Testing the Deployment

### Health Check

```bash
curl http://myprojectdemo.online/health
```

### Make a Prediction

Windows CMD:

```cmd
curl -X POST http://myprojectdemo.online/predict -H "Content-Type: application/json" -d "{\"age\": 55, \"sex\": 1, \"cp\": 3, \"trestbps\": 140, \"chol\": 250, \"fbs\": 0, \"restecg\": 1, \"thalach\": 150, \"exang\": 0, \"oldpeak\": 1.5, \"slope\": 2, \"ca\": 0, \"thal\": 3}"
```

Bash/Linux:

```bash
curl -X POST http://myprojectdemo.online/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 3, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
  }'
```

### View API Documentation

Open in browser: http://myprojectdemo.online/docs

---

## Monitoring and Logs

### View Container Logs

```bash
# API logs
docker logs -f heart-disease-api

# Nginx logs
docker logs -f nginx-proxy

# All logs
docker-compose logs -f
```

### Check Container Status

```bash
docker-compose ps
```

---

## Summary

Our deployment setup gives us:

- **Automated deployments** - Push to main and it deploys automatically
- **Zero-downtime updates** - Docker Compose handles container restarts
- **Easy rollback** - Can pull a previous image tag if needed
- **Centralized logs** - All container logs accessible via docker-compose

The whole deployment process from push to production takes about 6-10 minutes.
