"""
Prometheus metrics for Heart Disease Prediction API
Exposes metrics for Grafana Cloud integration
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

# Prediction metrics
prediction_counter = Counter(
    "heart_disease_predictions_total",
    "Total number of predictions made",
    ["prediction_result"],
)

prediction_latency = Histogram(
    "heart_disease_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)

# API request metrics
api_requests_total = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)

api_request_latency = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint", "method"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)

# Model metrics
model_loaded = Gauge(
    "heart_disease_model_loaded",
    "Whether the ML model is loaded (1=loaded, 0=not loaded)",
)

model_inference_time = Histogram(
    "heart_disease_model_inference_seconds",
    "Model inference time in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

# Error metrics
api_errors_total = Counter(
    "api_errors_total",
    "Total API errors",
    ["endpoint", "error_type"],
)

# Confidence metrics
prediction_confidence = Histogram(
    "heart_disease_prediction_confidence",
    "Prediction confidence scores",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
)


async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_prediction(prediction_result: int, confidence: float, latency: float):
    """Record prediction metrics"""
    result_label = "disease" if prediction_result == 1 else "healthy"
    prediction_counter.labels(prediction_result=result_label).inc()
    prediction_latency.observe(latency)
    prediction_confidence.observe(confidence)


def record_api_request(endpoint: str, method: str, status: int, latency: float):
    """Record API request metrics"""
    api_requests_total.labels(endpoint=endpoint, method=method, status=status).inc()
    api_request_latency.labels(endpoint=endpoint, method=method).observe(latency)


def record_api_error(endpoint: str, error_type: str):
    """Record API error metrics"""
    api_errors_total.labels(endpoint=endpoint, error_type=error_type).inc()


def set_model_loaded(loaded: bool):
    """Set model loaded status"""
    model_loaded.set(1 if loaded else 0)


def record_model_inference_time(inference_time: float):
    """Record model inference time"""
    model_inference_time.observe(inference_time)
