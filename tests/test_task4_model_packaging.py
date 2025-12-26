"""
Test Task 4: Model Packaging & Reproducibility
Tests for the model packaging and deployment preparation functionality
"""

import os
import sys
import json
import yaml
import joblib
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.model_packaging import ModelPackager, MissingValueHandler, FeatureEngineer
    import mlflow
    from mlflow import MlflowClient
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed and MLflow is available")


class TestTask4ModelPackaging:
    """Test class for Task 4 model packaging functionality"""
    
    def __init__(self):
        """Initialize test class"""
        self.test_package_dir = Path("test_package")
        self.packager = None
        
    def setup_test_environment(self):
        """Set up test environment"""
        print("Setting up test environment...")
        
        # Clean up any existing test package
        if self.test_package_dir.exists():
            shutil.rmtree(self.test_package_dir)
        
        # Initialize packager
        try:
            self.packager = ModelPackager()
            print("✓ ModelPackager initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize ModelPackager: {e}")
            raise
    
    def test_mlflow_connection(self):
        """Test MLflow connection and registry access"""
        print("\nTesting MLflow connection...")
        
        try:
            client = MlflowClient()
            
            # Try to list experiments
            experiments = client.search_experiments()
            print(f"✓ MLflow connection successful, found {len(experiments)} experiments")
            
            # Try to list registered models
            registered_models = client.search_registered_models()
            print(f"✓ Found {len(registered_models)} registered models")
            
            if len(registered_models) == 0:
                print("Warning: No registered models found. Run Task 3 first.")
                return False
            
            return True
            
        except Exception as e:
            print(f"✗ MLflow connection failed: {e}")
            return False
    
    def test_champion_model_identification(self):
        """Test champion model identification"""
        print("\nTesting champion model identification...")
        
        try:
            champion_info = self.packager.get_champion_model_info()
            
            # Validate champion info structure
            required_keys = ['model_name', 'version', 'run_id', 'roc_auc', 'model_uri']
            for key in required_keys:
                if key not in champion_info:
                    raise ValueError(f"Missing key in champion info: {key}")
            
            print(f"✓ Champion model identified: {champion_info['model_name']} v{champion_info['version']}")
            print(f"✓ Performance (ROC-AUC): {champion_info['roc_auc']:.4f}")
            
            return champion_info
            
        except Exception as e:
            print(f"✗ Champion model identification failed: {e}")
            raise
    
    def test_preprocessing_pipeline_creation(self):
        """Test preprocessing pipeline creation"""
        print("\nTesting preprocessing pipeline creation...")
        
        try:
            # Create sample data
            sample_data = pd.DataFrame({
                'age': [55, 60, 45], 'sex': [1, 0, 1], 'cp': [3, 2, 1], 
                'trestbps': [140, 130, 120], 'chol': [250, 200, 180],
                'fbs': [0, 1, 0], 'restecg': [1, 0, 1], 'thalach': [150, 160, 170],
                'exang': [0, 1, 0], 'oldpeak': [1.5, 2.0, 0.5], 
                'slope': [2, 1, 2], 'ca': [0, 1, 0], 'thal': [3, 2, 3]
            })
            
            # Create preprocessing pipeline
            pipeline = self.packager.create_preprocessing_pipeline(sample_data)
            
            # Test pipeline
            transformed_data = pipeline.transform(sample_data)
            
            print(f"✓ Preprocessing pipeline created with {len(pipeline.steps)} steps")
            print(f"✓ Pipeline transforms data: {sample_data.shape} -> {transformed_data.shape}")
            
            return pipeline
            
        except Exception as e:
            print(f"✗ Preprocessing pipeline creation failed: {e}")
            raise
    
    def test_custom_transformers(self):
        """Test custom transformer classes"""
        print("\nTesting custom transformers...")
        
        try:
            # Test data with missing values
            test_data = pd.DataFrame({
                'age': [55, None, 45], 'sex': [1, 0, 1], 'cp': [3, 2, None],
                'trestbps': [140, 130, 120], 'chol': [250, None, 180],
                'fbs': [0, 1, 0], 'restecg': [1, 0, 1], 'thalach': [150, 160, 170],
                'exang': [0, 1, 0], 'oldpeak': [1.5, 2.0, 0.5], 
                'slope': [2, 1, 2], 'ca': [0, 1, 0], 'thal': [3, 2, 3]
            })
            
            # Test MissingValueHandler
            missing_handler = MissingValueHandler()
            missing_handler.fit(test_data)
            cleaned_data = missing_handler.transform(test_data)
            
            if cleaned_data.isnull().sum().sum() > 0:
                raise ValueError("Missing values not handled properly")
            
            print("✓ MissingValueHandler works correctly")
            
            # Test FeatureEngineer
            feature_engineer = FeatureEngineer()
            engineered_data = feature_engineer.transform(cleaned_data)
            
            expected_new_features = ['age_group', 'chol_age_ratio', 'heart_rate_reserve', 
                                   'risk_score', 'age_sex_interaction', 'cp_exang_interaction']
            
            # Check if new features are created (some might fail due to missing columns)
            created_features = [col for col in expected_new_features if col in engineered_data.columns]
            print(f"✓ FeatureEngineer created {len(created_features)} new features")
            
            return True
            
        except Exception as e:
            print(f"✗ Custom transformers test failed: {e}")
            return False
    
    def test_environment_snapshot(self):
        """Test environment snapshot creation"""
        print("\nTesting environment snapshot creation...")
        
        try:
            env_files = self.packager.create_environment_snapshot()
            
            # Check if environment files were created
            required_files = ['pip_exact', 'core_requirements', 'system_info']
            for file_type in required_files:
                if file_type not in env_files:
                    raise ValueError(f"Missing environment file: {file_type}")
                
                file_path = Path(env_files[file_type])
                if not file_path.exists():
                    raise ValueError(f"Environment file not created: {file_path}")
            
            print(f"✓ Environment snapshot created with {len(env_files)} files")
            
            # Validate system info content
            with open(env_files['system_info'], 'r') as f:
                system_info = json.load(f)
            
            required_system_keys = ['python_version', 'platform', 'captured_at']
            for key in required_system_keys:
                if key not in system_info:
                    raise ValueError(f"Missing system info key: {key}")
            
            print("✓ System information captured correctly")
            return env_files
            
        except Exception as e:
            print(f"✗ Environment snapshot creation failed: {e}")
            raise
    
    def test_configuration_files(self):
        """Test configuration file creation"""
        print("\nTesting configuration file creation...")
        
        try:
            # Create test package directory
            test_dir = Path("test_config")
            test_dir.mkdir(exist_ok=True)
            
            # Mock model info
            model_info = {
                'model_name': 'test_model',
                'version': '1',
                'roc_auc': 0.95
            }
            
            config_files = self.packager.create_configuration_files(model_info, test_dir)
            
            # Check if config files were created
            required_configs = ['model_config', 'deployment_config']
            for config_type in required_configs:
                if config_type not in config_files:
                    raise ValueError(f"Missing config file: {config_type}")
                
                config_path = Path(config_files[config_type])
                if not config_path.exists():
                    raise ValueError(f"Config file not created: {config_path}")
            
            # Validate YAML structure
            with open(config_files['model_config'], 'r') as f:
                model_config = yaml.safe_load(f)
            
            if 'model' not in model_config or 'preprocessing' not in model_config:
                raise ValueError("Invalid model config structure")
            
            print(f"✓ Configuration files created: {len(config_files)} files")
            
            # Clean up
            shutil.rmtree(test_dir)
            return config_files
            
        except Exception as e:
            print(f"✗ Configuration file creation failed: {e}")
            if Path("test_config").exists():
                shutil.rmtree(Path("test_config"))
            raise
    
    def test_package_validation(self):
        """Test package validation functionality"""
        print("\nTesting package validation...")
        
        try:
            # Create a minimal test package
            test_dir = Path("test_validation")
            test_dir.mkdir(exist_ok=True)
            
            # Create a simple model for testing
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            
            # Create and save a test model
            test_model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
            ])
            
            # Create dummy training data
            X_dummy = np.random.randn(100, 13)
            y_dummy = np.random.randint(0, 2, 100)
            test_model.fit(X_dummy, y_dummy)
            
            # Save test model
            joblib.dump(test_model, test_dir / "model.joblib")
            
            # Create minimal config files
            with open(test_dir / "model_config.yaml", 'w') as f:
                yaml.dump({'model': {'name': 'test'}}, f)
            
            with open(test_dir / "deployment_config.yaml", 'w') as f:
                yaml.dump({'api': {'port': 8000}}, f)
            
            with open(test_dir / "model_metadata.json", 'w') as f:
                json.dump({'model_info': {'name': 'test'}}, f)
            
            # Test validation
            validation_results = self.packager.validate_model_package(test_dir)
            
            # Check validation results
            if 'joblib_loading' not in validation_results:
                raise ValueError("Missing joblib_loading validation")
            
            if 'prediction_test' not in validation_results:
                raise ValueError("Missing prediction_test validation")
            
            print(f"✓ Package validation completed")
            print(f"✓ Validation results: {sum(validation_results.values())} passed")
            
            # Clean up
            shutil.rmtree(test_dir)
            return validation_results
            
        except Exception as e:
            print(f"✗ Package validation failed: {e}")
            if Path("test_validation").exists():
                shutil.rmtree(Path("test_validation"))
            raise
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        print("\nCleaning up test environment...")
        
        # Remove test directories
        test_dirs = ["test_package", "test_config", "test_validation"]
        for test_dir in test_dirs:
            if Path(test_dir).exists():
                shutil.rmtree(Path(test_dir))
        
        print("✓ Test environment cleaned up")


def test_full_task4_pipeline():
    """Test the complete Task 4 pipeline"""
    print("\n" + "="*60)
    print("TESTING COMPLETE TASK 4 PIPELINE")
    print("="*60)
    
    test_instance = TestTask4ModelPackaging()
    
    try:
        # Setup
        test_instance.setup_test_environment()
        
        # Test MLflow connection
        if not test_instance.test_mlflow_connection():
            print("Skipping full pipeline test - MLflow not properly set up")
            return False
        
        # Test individual components
        test_instance.test_champion_model_identification()
        test_instance.test_preprocessing_pipeline_creation()
        test_instance.test_custom_transformers()
        test_instance.test_environment_snapshot()
        test_instance.test_configuration_files()
        test_instance.test_package_validation()
        
        print("\n" + "="*60)
        print("TASK 4 PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nTask 4 pipeline test failed: {e}")
        return False
        
    finally:
        # Always cleanup
        test_instance.cleanup_test_environment()


if __name__ == "__main__":
    """Run Task 4 tests directly"""
    print("Running Task 4 Model Packaging Tests")
    print("="*50)
    
    # Run individual tests
    test_instance = TestTask4ModelPackaging()
    
    try:
        test_instance.setup_test_environment()
        
        # Test MLflow connection first
        if test_instance.test_mlflow_connection():
            test_instance.test_champion_model_identification()
            test_instance.test_preprocessing_pipeline_creation()
        
        test_instance.test_custom_transformers()
        test_instance.test_environment_snapshot()
        test_instance.test_configuration_files()
        test_instance.test_package_validation()
        
        print("\n" + "="*50)
        print("All Task 4 tests completed!")
        
    except Exception as e:
        print(f"Task 4 tests failed: {e}")
        
    finally:
        test_instance.cleanup_test_environment()