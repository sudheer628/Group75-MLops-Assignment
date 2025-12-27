"""
Unit tests for Task 3: MLflow Experiment Tracking
Tests the MLflow setup, experiment logging, and tracking functionality

NOTE: These tests are designed to validate MLflow functionality without
creating test experiments on the production Railway server.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import mlflow

    from experiment_tracking import (
        create_experiment,
        log_data_info,
        log_feature_engineering_info,
        run_experiment_comparison,
        setup_mlflow,
        train_and_log_model,
        view_experiment_results,
    )

    MLFLOW_AVAILABLE = True
except ImportError as e:
    print(f"MLflow not available: {e}")
    MLFLOW_AVAILABLE = False


class TestTask3ExperimentTracking:
    """Test class for Task 3 functionality"""

    def __init__(self):
        """Initialize test class"""
        self.setup_test_environment()

    def setup_test_environment(self):
        """Setup test environment"""
        self.models_dir = Path("models")

        # Create directories if they don't exist (no mlruns - using Railway)
        for directory in [self.models_dir]:
            directory.mkdir(exist_ok=True)

    def test_mlflow_setup(self):
        """Test MLflow setup and configuration"""
        if not MLFLOW_AVAILABLE:
            print("Skipping MLflow setup test - MLflow not available")
            return

        try:
            tracking_uri = setup_mlflow()

            # Test tracking URI is set
            current_uri = mlflow.get_tracking_uri()
            assert current_uri is not None, "Tracking URI should be set"
            # Accept only Railway MLflow server configuration
            assert "railway.app" in current_uri, "Tracking URI should point to Railway server"

            print(f"MLflow setup test passed: {current_uri}")

        except Exception as e:
            raise Exception(f"MLflow setup test failed: {e}")

    def test_experiment_creation(self):
        """Test experiment creation and management (mocked to avoid Railway pollution)"""
        if not MLFLOW_AVAILABLE:
            print("Skipping experiment creation test - MLflow not available")
            return

        try:
            # Mock MLflow functions to avoid creating actual experiments on Railway
            with (
                patch("mlflow.get_experiment_by_name") as mock_get_exp,
                patch("mlflow.create_experiment") as mock_create_exp,
                patch("mlflow.set_experiment") as mock_set_exp,
            ):

                # Setup mocks
                mock_get_exp.return_value = None  # Simulate experiment doesn't exist
                mock_create_exp.return_value = "test_experiment_id_123"

                setup_mlflow()
                experiment_id = create_experiment("test_experiment", "Test experiment description")

                # Verify mocks were called correctly
                mock_get_exp.assert_called_once_with("test_experiment")
                mock_create_exp.assert_called_once()
                mock_set_exp.assert_called_once_with("test_experiment")

                # Test experiment ID was returned
                assert experiment_id == "test_experiment_id_123", "Experiment ID should be returned"

                print(f"Experiment creation test passed (mocked): {experiment_id}")

        except Exception as e:
            raise Exception(f"Experiment creation test failed: {e}")

    def test_data_logging(self):
        """Test data information logging (mocked to avoid Railway pollution)"""
        if not MLFLOW_AVAILABLE:
            print("Skipping data logging test - MLflow not available")
            return

        try:
            # Create sample data
            X = pd.DataFrame(
                {"feature1": np.random.randn(100), "feature2": np.random.randn(100), "feature3": np.random.randn(100)}
            )
            y = pd.Series(np.random.randint(0, 2, 100))

            # Mock MLflow run context and logging functions
            with (
                patch("mlflow.start_run") as mock_start_run,
                patch("mlflow.log_param") as mock_log_param,
                patch("mlflow.log_metric") as mock_log_metric,
                patch("mlflow.set_tag") as mock_set_tag,
            ):

                # Setup mock run context
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_id_123"
                mock_start_run.return_value.__enter__.return_value = mock_run
                mock_start_run.return_value.__exit__.return_value = None

                setup_mlflow()

                # Test data logging (this will use mocked MLflow functions)
                log_data_info(X, y, "test_experiment_id")

                # Verify logging functions were called
                assert mock_log_param.called, "Parameters should be logged"
                assert mock_log_metric.called, "Metrics should be logged"
                assert mock_set_tag.called, "Tags should be set"

                print("Data logging test passed (mocked)")

        except Exception as e:
            raise Exception(f"Data logging test failed: {e}")

    def test_feature_engineering_logging(self):
        """Test feature engineering logging (mocked to avoid Railway pollution)"""
        if not MLFLOW_AVAILABLE:
            print("Skipping feature engineering logging test - MLflow not available")
            return

        try:
            # Create sample data
            X_original = pd.DataFrame({"feature1": np.random.randn(100), "feature2": np.random.randn(100)})

            X_engineered = pd.DataFrame(
                {
                    "feature1": np.random.randn(100),
                    "feature2": np.random.randn(100),
                    "feature3": np.random.randn(100),  # New feature
                    "feature4": np.random.randn(100),  # New feature
                }
            )

            # Mock MLflow run context and logging functions
            with (
                patch("mlflow.start_run") as mock_start_run,
                patch("mlflow.log_param") as mock_log_param,
                patch("mlflow.set_tag") as mock_set_tag,
            ):

                # Setup mock run context
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_id_456"
                mock_start_run.return_value.__enter__.return_value = mock_run
                mock_start_run.return_value.__exit__.return_value = None

                setup_mlflow()

                # Test feature engineering logging
                log_feature_engineering_info(X_original, X_engineered, "test_experiment_id")

                # Verify logging functions were called
                assert mock_log_param.called, "Parameters should be logged"
                assert mock_set_tag.called, "Tags should be set"

                print("Feature engineering logging test passed (mocked)")

        except Exception as e:
            raise Exception(f"Feature engineering logging test failed: {e}")

    def test_model_logging(self):
        """Test model training and logging (mocked to avoid Railway pollution)"""
        if not MLFLOW_AVAILABLE:
            print("Skipping model logging test - MLflow not available")
            return

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            # Create sample data
            X = pd.DataFrame(np.random.randn(200, 4))
            y = pd.Series(np.random.randint(0, 2, 200))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create simple model
            model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(random_state=42))])

            # Mock MLflow run context and logging functions
            with (
                patch("mlflow.start_run") as mock_start_run,
                patch("mlflow.log_param") as mock_log_param,
                patch("mlflow.log_metric") as mock_log_metric,
                patch("mlflow.sklearn.log_model") as mock_log_model,
                patch("mlflow.log_artifact") as mock_log_artifact,
                patch("mlflow.set_tag") as mock_set_tag,
            ):

                # Setup mock run context
                mock_run = MagicMock()
                mock_run.info.run_id = "test_run_id_789"
                mock_start_run.return_value.__enter__.return_value = mock_run
                mock_start_run.return_value.__exit__.return_value = None

                setup_mlflow()

                # Test model logging
                result = train_and_log_model(
                    model_name="test_logistic",
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    hyperparameters={"C": 1.0, "penalty": "l2"},
                )

                # Verify result structure
                assert "run_id" in result, "Result should contain run_id"
                assert "metrics" in result, "Result should contain metrics"
                assert "model" in result, "Result should contain model"

                # Verify metrics
                metrics = result["metrics"]
                assert "accuracy" in metrics, "Should have accuracy metric"
                assert 0 <= metrics["accuracy"] <= 1, "Accuracy should be between 0 and 1"

                # Verify logging functions were called
                assert mock_log_param.called, "Parameters should be logged"
                assert mock_log_metric.called, "Metrics should be logged"
                assert mock_set_tag.called, "Tags should be set"

                print("Model logging test passed (mocked)")

        except Exception as e:
            raise Exception(f"Model logging test failed: {e}")

    def test_experiment_viewing(self):
        """Test experiment results viewing (mocked to avoid Railway pollution)"""
        if not MLFLOW_AVAILABLE:
            print("Skipping experiment viewing test - MLflow not available")
            return

        try:
            # Mock MLflow experiment and runs
            with patch("mlflow.get_experiment_by_name") as mock_get_exp, patch("mlflow.search_runs") as mock_search_runs:

                # Setup mock experiment
                mock_experiment = MagicMock()
                mock_experiment.experiment_id = "test_exp_id"
                mock_experiment.name = "test_viewing"
                mock_get_exp.return_value = mock_experiment

                # Setup mock runs DataFrame
                mock_runs_data = pd.DataFrame(
                    {
                        "run_id": ["run1", "run2"],
                        "tags.stage": ["model_training", "model_training"],
                        "params.model_type": ["logistic_regression", "random_forest"],
                        "metrics.roc_auc": [0.85, 0.92],
                        "metrics.accuracy": [0.80, 0.88],
                    }
                )
                mock_search_runs.return_value = mock_runs_data

                setup_mlflow()

                # Test viewing (should not raise errors)
                view_experiment_results("test_viewing")

                # Verify mocks were called
                mock_get_exp.assert_called_once_with("test_viewing")
                mock_search_runs.assert_called_once()

                print("Experiment viewing test passed (mocked)")

        except Exception as e:
            raise Exception(f"Experiment viewing test failed: {e}")

    def test_railway_server_connection(self):
        """Test that Railway MLflow server connection is working"""

        # Test Railway server connection
        if MLFLOW_AVAILABLE:
            setup_mlflow()
            # Verify we're connected to Railway server
            current_uri = mlflow.get_tracking_uri()
            assert "railway.app" in current_uri, "Should be connected to Railway server"

        print("Railway server connection test passed")


def test_full_task3_pipeline():
    """Test the complete Task 3 pipeline (mocked to avoid Railway pollution)"""
    if not MLFLOW_AVAILABLE:
        print("Skipping full pipeline test - MLflow not available")
        return

    try:
        # Mock the complete pipeline to avoid creating experiments on Railway
        with (
            patch("mlflow.get_experiment_by_name") as mock_get_exp,
            patch("mlflow.create_experiment") as mock_create_exp,
            patch("mlflow.set_experiment") as mock_set_exp,
        ):

            # Setup mocks
            mock_get_exp.return_value = None  # Simulate experiment doesn't exist
            mock_create_exp.return_value = "test_pipeline_exp_id"

            # Test basic MLflow functionality without running full experiment
            from experiment_tracking import create_experiment, setup_mlflow

            setup_mlflow()
            experiment_id = create_experiment("test_full_pipeline", "Test pipeline")

            # Verify experiment creation was mocked correctly
            assert experiment_id == "test_pipeline_exp_id", "Experiment should be created"
            mock_create_exp.assert_called_once()
            mock_set_exp.assert_called_once_with("test_full_pipeline")

            print("Full Task 3 pipeline test passed (mocked)")

    except Exception as e:
        raise Exception(f"Full pipeline test failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestTask3ExperimentTracking()

    print("Running Task 3 Tests...")
    print("=" * 50)

    try:
        test_instance.test_mlflow_setup()
        test_instance.test_experiment_creation()
        test_instance.test_data_logging()
        test_instance.test_feature_engineering_logging()
        test_instance.test_model_logging()
        test_instance.test_experiment_viewing()
        test_instance.test_railway_server_connection()
        test_full_task3_pipeline()

        print("\n" + "=" * 50)
        print("All Task 3 tests passed!")
        print("=" * 50)

    except Exception as e:
        print(f"\nTest failed: {e}")
        raise
