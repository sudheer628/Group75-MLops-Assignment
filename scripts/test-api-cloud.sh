#!/bin/bash

# Heart Disease API Cloud Testing Script
# For use in GitHub Codespaces or CI/CD environments
set -e

echo "Heart Disease API Cloud Testing"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="heart-disease-api"
CONTAINER_NAME="heart-disease-api-test"
PORT="8000"
WAIT_TIME=20

# Function to cleanup
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up containers...${NC}"
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}

# Trap cleanup on exit
trap cleanup EXIT

echo -e "${BLUE} Building Docker image...${NC}"
docker build -t $IMAGE_NAME .

echo -e "${BLUE} Starting container...${NC}"
docker run -d -p $PORT:$PORT --name $CONTAINER_NAME $IMAGE_NAME

echo -e "${BLUE} Waiting ${WAIT_TIME} seconds for startup...${NC}"
sleep $WAIT_TIME

echo -e "${BLUE} Testing API endpoints...${NC}"

# Test health endpoint
echo "Testing /health endpoint..."
health_response=$(curl -s -f http://localhost:$PORT/health)
if [ $? -eq 0 ]; then
    echo -e "${GREEN} Health check passed${NC}"
    echo "Response: $health_response"
else
    echo -e "${RED} Health check failed${NC}"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Test root endpoint
echo "Testing / endpoint..."
root_response=$(curl -s -f http://localhost:$PORT/)
if [ $? -eq 0 ]; then
    echo -e "${GREEN} Root endpoint passed${NC}"
else
    echo -e "${RED} Root endpoint failed${NC}"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Test prediction endpoint with high-risk sample
echo "Testing /predict endpoint (high-risk sample)..."
prediction_response=$(curl -s -X POST "http://localhost:$PORT/predict" \
  -H "Content-Type: application/json" \
  -d @test-data/sample-input.json)

if [ $? -eq 0 ]; then
    echo -e "${GREEN} Prediction endpoint passed${NC}"
    echo "Response: $prediction_response"
    
    # Validate response contains required fields
    if echo "$prediction_response" | jq -e '.prediction' > /dev/null && \
       echo "$prediction_response" | jq -e '.confidence' > /dev/null && \
       echo "$prediction_response" | jq -e '.probabilities' > /dev/null; then
        echo -e "${GREEN} Response format validated${NC}"
    else
        echo -e "${RED} Invalid response format${NC}"
        docker logs $CONTAINER_NAME
        exit 1
    fi
else
    echo -e "${RED} Prediction endpoint failed${NC}"
    docker logs $CONTAINER_name
    exit 1
fi

# Test prediction endpoint with healthy sample
echo "Testing /predict endpoint (healthy sample)..."
healthy_response=$(curl -s -X POST "http://localhost:$PORT/predict" \
  -H "Content-Type: application/json" \
  -d @test-data/sample-input-healthy.json)

if [ $? -eq 0 ]; then
    echo -e "${GREEN} Healthy sample prediction passed${NC}"
    echo "Response: $healthy_response"
else
    echo -e "${RED} Healthy sample prediction failed${NC}"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Test model info endpoint
echo "Testing /model/info endpoint..."
model_info_response=$(curl -s -f http://localhost:$PORT/model/info)
if [ $? -eq 0 ]; then
    echo -e "${GREEN} Model info endpoint passed${NC}"
    echo "Response: $model_info_response"
else
    echo -e "${RED} Model info endpoint failed${NC}"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Test API documentation endpoints
echo "Testing /docs endpoint..."
if curl -s -f http://localhost:$PORT/docs > /dev/null; then
    echo -e "${GREEN} API documentation accessible${NC}"
else
    echo -e "${RED} API documentation failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN} All tests passed successfully!${NC}"
echo "=================================="
echo -e "${BLUE} Test Summary:${NC}"
echo "- Container build and startup"
echo "- Health check endpoint"
echo "- Root endpoint"
echo "- Prediction endpoint (high-risk)"
echo "- Prediction endpoint (low-risk)"
echo "- Model info endpoint"
echo "- API documentation"
echo ""
echo -e "${BLUE} API Access:${NC}"
echo "- API Base: http://localhost:$PORT"
echo "- Documentation: http://localhost:$PORT/docs"
echo "- Health Check: http://localhost:$PORT/health"
echo ""
echo -e "${YELLOW} In GitHub Codespaces, the API will be automatically forwarded and accessible via the forwarded port.${NC}"