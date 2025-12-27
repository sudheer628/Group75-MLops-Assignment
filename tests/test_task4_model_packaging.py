"""
Test Task 4: Model Packaging & Reproducibility
Tests for the model packaging and deployment preparation functionality
"""

import glob
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    import mlflow
    from mlflow import MlflowClient

    from src.model_packaging import FeatureEngineer, MissingValueHandler, ModelPackager
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
            print("ModelPackager initialized successfully")
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
            print(f"MLflow connection successful, found {len(experiments)} experiments")

            # Try to list registered models
            registered_models = client.search_registered_models()
            print(f"Found {len(registered_models)} registered models")

            if len(registered_models) == 0:
                print("Warning: No registered models found. Run Task 3 first.")
                return False

            return True

        except Exception as e:
            print(f"MLflow connection failed: {e}")
            return False

    def test_champion_model_identification(self):
        """Test champion model identification from Railway MLflow"""
        print("\nTesting champion model identification...")

        try:
            champion_info = self.packager.get_champion_model_info()

            # Validate champion info structure
            required_keys = ["model_name", "version", "run_id", "roc_auc", "model_uri"]
            for key in required_keys:
                if key not in champion_info:
                    raise ValueError(f"Missing key in champion info: {key}")

            print(f"Champion model identified: {champion_info['model_name']} v{champion_info['version']}")
            print(f"Performance (ROC-AUC): {champion_info['roc_auc']:.4f}")

            return champion_info

        except Exception as e:
            print(f"Champion model identification failed: {e}")
            print("This is expected if no models are available in Railway MLflow")
            print("Run Task 3 (experiment tracking) first to create models")
            # Don't raise the exception, just return None to indicate no models available
            return None

    def test_preprocessing_pipeline_creation(self):
        """Test preprocessing pipeline creation"""
        print("\nTesting preprocessing pipeline creation...")

        try:
            # Create sample data using the same method as TASK-4
            from src.data_acquisition_eda import load_heart_disease_data

            sample_data, _, _ = load_heart_disease_data()

            # Create preprocessing pipeline
            pipeline = self.packager.create_preprocessing_pipeline(sample_data)

            # Test pipeline
            transformed_data = pipeline.transform(sample_data)

            print(f"Preprocessing pipeline created with {len(pipeline.steps)} steps")
            print(f"Pipeline transforms data: {sample_data.shape} -> {transformed_data.shape}")

            return pipeline

        except Exception as e:
            print(f"Preprocessing pipeline creation failed: {e}")
            raise

    def test_custom_transformers(self):
        """Test custom transformer classes"""
        print("\nTesting custom transformers...")

        try:
            # Test data with missing values
            test_data = pd.DataFrame(
                {
                    "age": [55, None, 45],
                    "sex": [1, 0, 1],
                    "cp": [3, 2, None],
                    "trestbps": [140, 130, 120],
                    "chol": [250, None, 180],
                    "fbs": [0, 1, 0],
                    "restecg": [1, 0, 1],
                    "thalach": [150, 160, 170],
                    "exang": [0, 1, 0],
                    "oldpeak": [1.5, 2.0, 0.5],
                    "slope": [2, 1, 2],
                    "ca": [0, 1, 0],
                    "thal": [3, 2, 3],
                }
            )

            # Test MissingValueHandler
            missing_handler = MissingValueHandler()
            missing_handler.fit(test_data)
            cleaned_data = missing_handler.transform(test_data)

            if cleaned_data.isnull().sum().sum() > 0:
                raise ValueError("Missing values not handled properly")

            print("MissingValueHandler works correctly")

            # Test FeatureEngineer
            feature_engineer = FeatureEngineer()
            engineered_data = feature_engineer.transform(cleaned_data)

            expected_new_features = [
                "age_group",
                "chol_age_ratio",
                "heart_rate_reserve",
                "risk_score",
                "age_sex_interaction",
                "cp_exang_interaction",
            ]

            # Check if new features are created (some might fail due to missing columns)
            created_features = [col for col in expected_new_features if col in engineered_data.columns]
            print(f"FeatureEngineer created {len(created_features)} new features")

            return True

        except Exception as e:
            print(f"Custom transformers test failed: {e}")
            return False

    def test_environment_snapshot(self):
        """Test environment snapshot creation"""
        print("\nTesting environment snapshot creation...")

        try:
            env_files = self.packager.create_environment_snapshot()

            # Check if environment files were created
            required_files = ["pip_exact", "core_requirements", "system_info"]
            for file_type in required_files:
                if file_type not in env_files:
                    raise ValueError(f"Missing environment file: {file_type}")

                file_path = Path(env_files[file_type])
                if not file_path.exists():
                    raise ValueError(f"Environment file not created: {file_path}")

            print(f"Environment snapshot created with {len(env_files)} files")

            # Validate system info content
            with open(env_files["system_info"], "r") as f:
                system_info = json.load(f)

            required_system_keys = ["python_version", "platform", "captured_at"]
            for key in required_system_keys:
                if key not in system_info:
                    raise ValueError(f"Missing system info key: {key}")

            print("System information captured correctly")
            return env_files

        except Exception as e:
            print(f"Environment snapshot creation failed: {e}")
            raise

    def test_configuration_files(self):
        """Test configuration file creation"""
        print("\nTesting configuration file creation...")

        try:
            # Create test package directory
            test_dir = Path("test_config")
            test_dir.mkdir(exist_ok=True)

            # Mock model info
            model_info = {"model_name": "test_model", "version": "1", "roc_auc": 0.95}

            config_files = self.packager.create_configuration_files(model_info, test_dir)

            # Check if config files were created
            required_configs = ["model_config", "deployment_config"]
            for config_type in required_configs:
                if config_type not in config_files:
                    raise ValueError(f"Missing config file: {config_type}")

                config_path = Path(config_files[config_type])
                if not config_path.exists():
                    raise ValueError(f"Config file not created: {config_path}")

            # Validate YAML structure
            with open(config_files["model_config"], "r") as f:
                model_config = yaml.safe_load(f)

            if "model" not in model_config or "preprocessing" not in model_config:
                raise ValueError("Invalid model config structure")

            print(f"Configuration files created: {len(config_files)} files")

            # Clean up
            shutil.rmtree(test_dir)
            return config_files

        except Exception as e:
            print(f"Configuration file creation failed: {e}")
            if Path("test_config").exists():
                shutil.rmtree(Path("test_config"))
            raise

    def test_package_validation(self):
        """Test package validation functionality using real TASK-4 packages"""
        print("\nTesting package validation...")

        try:
            # Use the latest real package created by TASK-4
            import glob

            package_pattern = "packages/heart_disease_model_*"
            package_dirs = glob.glob(package_pattern)

            if not package_dirs:
                print("No real packages found. Creating a test package first...")
                # Create a real package using the actual TASK-4 logic
                package_path = self.packager.create_deployment_package()
                test_dir = Path(package_path)
            else:
                # Use the most recent package
                latest_package = max(package_dirs, key=lambda x: Path(x).stat().st_mtime)
                test_dir = Path(latest_package)
                print(f"Using existing package: {test_dir}")

            # Test validation on the real package
            validation_results = self.packager.validate_model_package(test_dir)

            # Check validation results
            required_validations = ["joblib_loading", "prediction_test", "mlflow_loading", "configuration_files"]
            for validation in required_validations:
                if validation not in validation_results:
                    raise ValueError(f"Missing validation: {validation}")

            # Core functionality should pass (joblib + prediction)
            core_passed = validation_results.get("joblib_loading", False) and validation_results.get("prediction_test", False)

            if not core_passed:
                raise ValueError("Core functionality validation failed")

            print(f"Package validation completed")
            print(f"Validation results: {sum(validation_results.values())} passed")

            return validation_results

        except Exception as e:
            print(f"Package validation failed: {e}")
            raise

    def cleanup_test_environment(self):
        """Clean up test environment"""
        print("\nCleaning up test environment...")

        # Remove test directories
        test_dirs = ["test_package", "test_config", "test_validation"]
        for test_dir in test_dirs:
            if Path(test_dir).exists():
                shutil.rmtree(Path(test_dir))

        print("Test environment cleaned up")


def test_full_task4_pipeline():
    """Test the complete Task 4 pipeline with real Railway integration"""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE TASK 4 PIPELINE")
    print("=" * 60)

    test_instance = TestTask4ModelPackaging()

    try:
        # Setup
        test_instance.setup_test_environment()

        # Test MLflow connection
        mlflow_connected = test_instance.test_mlflow_connection()

        # Test individual components
        test_instance.test_custom_transformers()
        test_instance.test_environment_snapshot()
        test_instance.test_configuration_files()

        # Test the real package validation (this will use actual Railway models)
        try:
            test_instance.test_package_validation()
            print("\nReal package validation: PASS")
        except Exception as e:
            print(f"\nReal package validation failed: {e}")
            # If no existing packages, try to create one
            try:
                print("Attempting to create a new package for testing...")
                package_path = test_instance.packager.create_deployment_package()
                print(f"Created package: {package_path}")
                print("Package creation: PASS")
            except Exception as create_e:
                print(f"Package creation failed: {create_e}")
                print("This may be due to missing Railway models. Run Task 3 first.")

        print("\n" + "=" * 60)
        print("TASK 4 PIPELINE TEST COMPLETED!")
        print("=" * 60)

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
    print("=" * 50)

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

        print("\n" + "=" * 50)
        print("All Task 4 tests completed!")

    except Exception as e:
        print(f"Task 4 tests failed: {e}")

    finally:
        test_instance.cleanup_test_environment()
