# 🚀 GitHub Codespaces Quick Start Guide

## Heart Disease MLOps Project - Cloud Development

This guide helps you get started with the Heart Disease MLOps project using GitHub Codespaces - no local Docker installation required!

## 🌟 **Option 2: GitHub Actions Testing (Recommended)**

### **Automatic Testing on Every Push**

1. **Push your changes** to GitHub
2. **GitHub Actions automatically**:
   - Builds the Docker container
   - Tests all API endpoints
   - Pushes to GitHub Container Registry
   - Provides detailed test results

### **View Results**

- Go to **Actions** tab in your GitHub repository
- Check the **"Build and Push Container"** workflow
- View detailed test results and container build logs

---

## 🌟 **Option 3: GitHub Codespaces Development**

### **1. Launch Codespaces**

Click the **"Code"** button in your GitHub repository → **"Codespaces"** → **"Create codespace on main"**

### **2. Automatic Setup**

Codespaces will automatically:

- ✅ Install Python 3.11
- ✅ Install Docker
- ✅ Install all project dependencies
- ✅ Configure VS Code extensions
- ✅ Set up development environment

### **3. Test the API**

```bash
# Run the cloud testing script
chmod +x scripts/test-api-cloud.sh
./scripts/test-api-cloud.sh
```

### **4. Development Workflow**

```bash
# Build and run the container
docker build -t heart-disease-api .
docker run -d -p 8000:8000 --name api heart-disease-api

# The API will be automatically forwarded
# Access via the "Ports" tab in VS Code
```

### **5. Access the API**

- **Automatic Port Forwarding**: Codespaces automatically forwards port 8000
- **Click the "Ports" tab** in VS Code terminal
- **Open the forwarded URL** to access your API
- **API Documentation**: Add `/docs` to the URL for Swagger UI

---

## 🧪 **Testing Endpoints**

### **Health Check**

```bash
curl https://your-codespace-url/health
```

### **Prediction (High Risk)**

```bash
curl -X POST "https://your-codespace-url/predict" \
  -H "Content-Type: application/json" \
  -d @test-data/sample-input.json
```

### **Prediction (Low Risk)**

```bash
curl -X POST "https://your-codespace-url/predict" \
  -H "Content-Type: application/json" \
  -d @test-data/sample-input-healthy.json
```

### **API Documentation**

Open: `https://your-codespace-url/docs`

---

## 📊 **Expected API Response**

```json
{
  "prediction": 1,
  "confidence": 0.85,
  "probabilities": [0.15, 0.85],
  "risk_level": "High"
}
```

---

## 🔧 **Development Tips**

### **VS Code Extensions (Pre-installed)**

- Python support with linting and formatting
- Docker extension for container management
- YAML and JSON support
- Jupyter notebook support

### **Useful Commands**

```bash
# View running containers
docker ps

# View container logs
docker logs heart-disease-api

# Stop and remove container
docker stop heart-disease-api
docker rm heart-disease-api

# Rebuild after changes
docker build -t heart-disease-api .
```

### **File Watching**

- Changes to Python files require container rebuild
- Use Docker Compose for development with volume mounts:

```bash
docker-compose up -d
```

---

## 🚀 **Container Registry**

Your containers are automatically built and pushed to:

- `ghcr.io/your-username/your-repo/heart-disease-api:latest`
- `ghcr.io/your-username/your-repo/heart-disease-api:main-sha123`

### **Pull and Run from Registry**

```bash
# Pull the latest image
docker pull ghcr.io/your-username/your-repo/heart-disease-api:latest

# Run from registry
docker run -p 8000:8000 ghcr.io/your-username/your-repo/heart-disease-api:latest
```

---

## 🎯 **Task 6 Validation**

### **Requirements Checklist**

- ✅ Docker container builds successfully
- ✅ FastAPI application with /predict endpoint
- ✅ Accepts JSON input with validation
- ✅ Returns prediction + confidence
- ✅ Container runs and responds to requests
- ✅ Sample input testing works

### **Bonus Features**

- ✅ GitHub Container Registry integration
- ✅ Automated CI/CD testing
- ✅ Multi-architecture builds
- ✅ Comprehensive API documentation
- ✅ Health checks and monitoring
- ✅ Production-ready security

---

## 🆘 **Troubleshooting**

### **Container Won't Start**

```bash
# Check container logs
docker logs heart-disease-api

# Check if models exist
ls -la models/
```

### **API Not Responding**

```bash
# Check if container is running
docker ps

# Check port forwarding in Codespaces
# Go to "Ports" tab in VS Code
```

### **Model Loading Issues**

```bash
# Verify model files exist
ls -la models/best_model.joblib

# Check container logs for model loading errors
docker logs heart-disease-api
```

---

## 🎉 **Success!**

If all tests pass, your Task 6 implementation is complete and ready for Task 7 (Production Deployment)!

**Next Steps:**

- Proceed to Task 7 for Kubernetes deployment
- Set up monitoring and logging (Task 8)
- Create documentation and reporting (Task 9)
