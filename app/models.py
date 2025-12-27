"""
Pydantic models for API input/output validation
"""
from pydantic import BaseModel, Field
from typing import List


class PredictionInput(BaseModel):
    """Input model for heart disease prediction"""
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0: female, 1: male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0: false, 1: true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0: no, 1: yes)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)")

    class Config:
        schema_extra = {
            "example": {
                "age": 55,
                "sex": 1,
                "cp": 3,
                "trestbps": 140,
                "chol": 250,
                "fbs": 0,
                "restecg": 1,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.5,
                "slope": 2,
                "ca": 0,
                "thal": 3
            }
        }


class PredictionOutput(BaseModel):
    """Output model for heart disease prediction"""
    prediction: int = Field(..., description="Predicted class (0: no disease, 1: disease)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    probabilities: List[float] = Field(..., description="Class probabilities [no_disease, disease]")
    risk_level: str = Field(..., description="Risk level (Low, Medium, High)")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "confidence": 0.85,
                "probabilities": [0.15, 0.85],
                "risk_level": "High"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: str = Field(..., description="Detailed error information")