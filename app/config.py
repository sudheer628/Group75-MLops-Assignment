"""
Configuration settings for the Heart Disease Prediction API
"""
import os
from pathlib import Path


class Settings:
    """Application settings"""
    
    # API Configuration
    API_TITLE = "Heart Disease Prediction API"
    API_DESCRIPTION = """
    A machine learning API for predicting heart disease risk based on clinical features.
    
    This API uses a trained machine learning model to predict the likelihood of heart disease
    based on 13 clinical features including age, sex, chest pain type, blood pressure, and more.
    """
    API_VERSION = "1.0.0"
    
    # Model Configuration
    MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
    PREPROCESSING_PIPELINE_PATH = os.getenv("PREPROCESSING_PIPELINE_PATH", "models/preprocessing_pipeline.joblib")
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    DEBUG = ENVIRONMENT == "development"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # CORS
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    
    @property
    def model_exists(self) -> bool:
        """Check if model file exists"""
        return Path(self.MODEL_PATH).exists()
    
    @property
    def preprocessing_pipeline_exists(self) -> bool:
        """Check if preprocessing pipeline exists"""
        return Path(self.PREPROCESSING_PIPELINE_PATH).exists()


settings = Settings()