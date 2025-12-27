"""
Heart disease prediction logic
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

from .config import settings
from .models import PredictionInput, PredictionOutput

logger = logging.getLogger(__name__)


class HeartDiseasePredictor:
    """Heart disease prediction model wrapper"""
    
    def __init__(self):
        self.model = None
        self.preprocessing_pipeline = None
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure the predictor is initialized (lazy loading)"""
        if not self._initialized:
            self.load_model()
            self._initialized = True
    
    def load_model(self) -> None:
        """Load the trained model and preprocessing pipeline"""
        try:
            # First, try to ensure model is available (download if necessary)
            from .model_loader import ProductionModelLoader
            
            loader = ProductionModelLoader()
            model_available = loader.ensure_model_available(settings.MODEL_PATH)
            
            if not model_available:
                logger.error("Failed to ensure model availability")
                raise FileNotFoundError(f"Could not load or download model: {settings.MODEL_PATH}")
            
            # Load main model
            if Path(settings.MODEL_PATH).exists():
                self.model = joblib.load(settings.MODEL_PATH)
                logger.info(f"Model loaded from {settings.MODEL_PATH}")
            else:
                logger.error(f"Model file not found: {settings.MODEL_PATH}")
                logger.info("Available files in models directory:")
                models_dir = Path("models")
                if models_dir.exists():
                    for file in models_dir.glob("*.joblib"):
                        logger.info(f"  - {file}")
                else:
                    logger.error("Models directory does not exist")
                raise FileNotFoundError(f"Model file not found: {settings.MODEL_PATH}")
            
            # Try to load preprocessing pipeline (optional for some models)
            if Path(settings.PREPROCESSING_PIPELINE_PATH).exists():
                self.preprocessing_pipeline = joblib.load(settings.PREPROCESSING_PIPELINE_PATH)
                logger.info(f"Preprocessing pipeline loaded from {settings.PREPROCESSING_PIPELINE_PATH}")
            else:
                logger.warning(f"Preprocessing pipeline not found: {settings.PREPROCESSING_PIPELINE_PATH}")
                logger.info("Using model without separate preprocessing pipeline")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, input_data: PredictionInput) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        # Convert input to DataFrame
        data_dict = input_data.dict()
        df = pd.DataFrame([data_dict])
        
        # Ensure correct column order
        df = df[self.feature_names]
        
        # Apply preprocessing pipeline if available
        if self.preprocessing_pipeline is not None:
            try:
                df_processed = self.preprocessing_pipeline.transform(df)
                # Convert back to DataFrame if it's a numpy array
                if isinstance(df_processed, np.ndarray):
                    df_processed = pd.DataFrame(df_processed, columns=self.feature_names)
                return df_processed
            except Exception as e:
                logger.warning(f"Preprocessing pipeline failed: {e}")
                logger.info("Using raw input data")
        
        return df
    
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Make prediction for heart disease"""
        try:
            # Ensure predictor is initialized
            self._ensure_initialized()
            
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)[0].tolist()
                confidence = max(probabilities)
            else:
                # For models without predict_proba, use decision function or default
                if hasattr(self.model, 'decision_function'):
                    decision_score = self.model.decision_function(processed_data)[0]
                    # Convert decision score to probability-like confidence
                    confidence = 1 / (1 + np.exp(-abs(decision_score)))
                    probabilities = [1 - confidence, confidence] if prediction == 1 else [confidence, 1 - confidence]
                else:
                    # Default confidence for models without probability estimates
                    confidence = 0.75
                    probabilities = [1 - confidence, confidence] if prediction == 1 else [confidence, 1 - confidence]
            
            # Determine risk level
            risk_level = self._get_risk_level(confidence, prediction)
            
            return PredictionOutput(
                prediction=int(prediction),
                confidence=float(confidence),
                probabilities=probabilities,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _get_risk_level(self, confidence: float, prediction: int) -> str:
        """Determine risk level based on prediction and confidence"""
        if prediction == 0:
            return "Low"
        else:
            if confidence >= 0.8:
                return "High"
            elif confidence >= 0.6:
                return "Medium"
            else:
                return "Low"
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        if not self._initialized:
            try:
                self._ensure_initialized()
            except Exception:
                return False
        return self.model is not None


# Global predictor instance
predictor = HeartDiseasePredictor()