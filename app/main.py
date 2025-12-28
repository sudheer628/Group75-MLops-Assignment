"""
Heart Disease Prediction FastAPI Application
"""
import logging
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .models import PredictionInput, PredictionOutput, HealthResponse, ErrorResponse
from .prediction import predictor
from .metrics import (
    metrics_endpoint,
    record_prediction,
    record_api_request,
    record_api_error,
    set_model_loaded,
    record_model_inference_time,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Starting Heart Disease Prediction API")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info("Model will be loaded on first prediction request (lazy loading)")

    # Create models directory if it doesn't exist
    from pathlib import Path

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down Heart Disease Prediction API")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Heart Disease Prediction API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "unhealthy",
        version=settings.API_VERSION,
        ml_model_loaded=predictor.is_loaded,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint for Grafana Cloud"""
    return await metrics_endpoint()


@app.post("/predict", response_model=PredictionOutput)
async def predict_heart_disease(input_data: PredictionInput):
    """
    Predict heart disease risk based on clinical features

    Returns prediction with confidence score and risk level.
    """
    start_time = time.time()

    try:
        if not predictor.is_loaded:
            record_api_error("/predict", "model_not_loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Service unavailable.",
            )

        # Make prediction
        result = predictor.predict(input_data)

        # Record metrics
        latency = time.time() - start_time
        record_prediction(result.prediction, result.confidence, latency)
        record_api_request("/predict", "POST", 200, latency)

        logger.info(
            f"Prediction made: {result.prediction} (confidence: {result.confidence:.3f})"
        )

        return result

    except HTTPException:
        latency = time.time() - start_time
        record_api_request("/predict", "POST", 503, latency)
        raise
    except Exception as e:
        latency = time.time() - start_time
        record_api_request("/predict", "POST", 500, latency)
        record_api_error("/predict", type(e).__name__)
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_loaded": predictor.is_loaded,
        "model_path": settings.MODEL_PATH,
        "preprocessing_pipeline_path": settings.PREPROCESSING_PIPELINE_PATH,
        "raw_feature_names": predictor.raw_feature_names,
        "engineered_feature_names": predictor.engineered_feature_names,
        "model_type": type(predictor.model).__name__ if predictor.model else None
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )