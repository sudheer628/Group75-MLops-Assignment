"""
Simplified model loader for Python 3.10 compatibility
"""
import logging
from pathlib import Path

import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

logger = logging.getLogger(__name__)


class ProductionModelLoader:
    """Simplified model loader for Railway MLflow server"""
    
    def __init__(self):
        self.railway_mlflow_uri = "https://mlflow-tracking-production-53fb.up.railway.app"
        mlflow.set_tracking_uri(self.railway_mlflow_uri)
        self.client = MlflowClient()
        
    def download_champion_model(self, local_model_path: str = "models/best_model.joblib") -> bool:
        """Download the champion model from Railway MLflow server"""
        try:
            logger.info("=== Model Download Process Starting ===")
            logger.info(f"Python version: {__import__('sys').version}")
            logger.info(f"Numpy version: {np.__version__}")
            logger.info(f"MLflow version: {mlflow.__version__}")
            logger.info("Connecting to Railway MLflow server...")
            
            # Get the experiment
            experiment = self.client.get_experiment_by_name("heart_disease_comparison")
            if not experiment:
                logger.error("Experiment not found")
                return False
                
            # Get all runs
            runs = self.client.search_runs(experiment_ids=[experiment.experiment_id], max_results=50)
            if not runs:
                logger.error("No runs found")
                return False
                
            # Find best model
            champion_info = None
            best_performance = 0
            
            for run in runs:
                if run.data.tags.get("stage") == "model_training":
                    roc_auc = run.data.metrics.get("roc_auc", 0)
                    if roc_auc > best_performance:
                        best_performance = roc_auc
                        champion_info = {
                            "run_id": run.info.run_id,
                            "model_uri": f"runs:/{run.info.run_id}/model"
                        }
            
            if not champion_info:
                logger.error("No champion model found")
                return False
                
            logger.info(f"Champion model ROC-AUC: {best_performance:.4f}")
            logger.info(f"Model URI: {champion_info['model_uri']}")
            
            # Create directory
            Path(local_model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download model using basic MLflow loading
            logger.info("Loading model from MLflow...")
            model = mlflow.sklearn.load_model(champion_info['model_uri'])
            logger.info(f"Model loaded successfully: {type(model).__name__}")
            
            # Save using joblib
            import joblib
            logger.info(f"Saving model to: {local_model_path}")
            joblib.dump(model, local_model_path)
            
            logger.info("=== Model Download Process Completed Successfully ===")
            return True
            
        except Exception as e:
            logger.error(f"=== Model Download Process Failed ===")
            logger.error(f"Error: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
            
    def ensure_model_available(self, model_path: str = "models/best_model.joblib") -> bool:
        """Ensure model is available locally"""
        if Path(model_path).exists():
            logger.info(f"Model exists: {model_path}")
            return True
            
        logger.info("Downloading model from Railway...")
        return self.download_champion_model(model_path)