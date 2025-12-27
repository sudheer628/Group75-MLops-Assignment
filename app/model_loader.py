"""
Production model loader for downloading models from Railway MLflow server
"""
import os
import logging
import warnings
from pathlib import Path
from typing import Optional

# Suppress Pydantic model namespace warnings
warnings.filterwarnings("ignore", message=".*Field.*has conflict with protected namespace.*")

# Handle numpy import issues
try:
    import numpy as np
except ImportError as e:
    logging.error(f"Numpy import failed: {e}")
    raise

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

logger = logging.getLogger(__name__)


class ProductionModelLoader:
    """Load models from Railway MLflow server for production deployment"""
    
    def __init__(self):
        # Set Railway MLflow server as tracking URI
        self.railway_mlflow_uri = "https://mlflow-tracking-production-53fb.up.railway.app"
        mlflow.set_tracking_uri(self.railway_mlflow_uri)
        self.client = MlflowClient()
        
    def download_champion_model(self, local_model_path: str = "models/best_model.joblib") -> bool:
        """
        Download the champion model from Railway MLflow server
        
        Args:
            local_model_path: Local path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Connecting to Railway MLflow server...")
            logger.info(f"MLflow URI: {self.railway_mlflow_uri}")
            
            # Get the heart disease comparison experiment
            experiment = self.client.get_experiment_by_name("heart_disease_comparison")
            if not experiment:
                logger.error("Experiment 'heart_disease_comparison' not found on Railway server")
                return False
                
            logger.info(f"Found experiment: {experiment.name}")
            
            # Get all runs from the experiment
            runs = self.client.search_runs(experiment_ids=[experiment.experiment_id], max_results=50)
            
            if not runs:
                logger.error("No runs found in the experiment")
                return False
                
            logger.info(f"Found {len(runs)} runs in experiment")
            
            # Find the best model from training runs
            champion_info = None
            best_performance = 0
            
            for run in runs:
                stage = run.data.tags.get("stage", "")
                if stage == "model_training":
                    model_type = run.data.params.get("model_type", "unknown")
                    roc_auc = run.data.metrics.get("roc_auc", 0)
                    
                    logger.info(f"Found model: {model_type}, ROC-AUC: {roc_auc:.4f}")
                    
                    if roc_auc > best_performance:
                        best_performance = roc_auc
                        champion_info = {
                            "model_type": f"heart_disease_{model_type}",
                            "run_id": run.info.run_id,
                            "roc_auc": roc_auc,
                            "model_uri": f"runs:/{run.info.run_id}/model",
                        }
            
            if not champion_info:
                logger.error("Could not identify champion model from training runs")
                return False
                
            logger.info(f"Champion model identified: {champion_info['model_type']}")
            logger.info(f"ROC-AUC: {champion_info['roc_auc']:.4f}")
            logger.info(f"Run ID: {champion_info['run_id']}")
            
            # Download the model
            logger.info(f"Downloading model from: {champion_info['model_uri']}")
            
            # Create models directory if it doesn't exist
            Path(local_model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download with retry logic for Railway
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    import time
                    
                    # Workaround for numpy._core issue
                    import numpy as np
                    # Force numpy to initialize properly
                    _ = np.array([1, 2, 3])
                    
                    # Load model from MLflow
                    model = mlflow.sklearn.load_model(champion_info['model_uri'])
                    
                    # Save model locally using joblib
                    import joblib
                    joblib.dump(model, local_model_path)
                    
                    logger.info(f"Model successfully downloaded and saved to: {local_model_path}")
                    
                    # Verify the model file exists and is valid
                    if Path(local_model_path).exists():
                        # Test loading the saved model
                        test_model = joblib.load(local_model_path)
                        logger.info("Model verification successful")
                        return True
                    else:
                        logger.error("Model file was not created successfully")
                        return False
                        
                except Exception as download_e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Download attempt {attempt + 1} failed: {download_e}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"All download attempts failed: {download_e}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error downloading champion model: {e}")
            return False
            
    def ensure_model_available(self, model_path: str = "models/best_model.joblib") -> bool:
        """
        Ensure model is available locally, download if necessary
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model is available, False otherwise
        """
        # Check if model already exists locally
        if Path(model_path).exists():
            logger.info(f"Model already exists locally: {model_path}")
            return True
            
        # Try to download from Railway
        logger.info("Model not found locally, attempting to download from Railway MLflow server...")
        return self.download_champion_model(model_path)


def download_model_for_container():
    """
    Standalone function to download model for container deployment
    This can be called during container startup or build process
    """
    loader = ProductionModelLoader()
    
    # Try to download the model
    success = loader.ensure_model_available()
    
    if success:
        print("Model successfully downloaded from Railway MLflow server")
        return True
    else:
        print("Failed to download model from Railway MLflow server")
        return False


if __name__ == "__main__":
    # Allow running this script directly to download models
    download_model_for_container()