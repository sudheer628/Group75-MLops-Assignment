"""
Unit tests for Task 2: Feature Engineering & Model Development
Tests the feature engineering, model training, and evaluation functionality
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import (
    create_preprocessing_pipeline,
    define_models,
    engineer_features,
    evaluate_models,
    handle_missing_values,
    load_processed_data,
    perform_cross_validation,
    prepare_target_variable,
    split_data,
)


class TestTask2FeatureEngineering:
    """Test class for Task 2 functionality"""

    def __init__(self):
        """Initialize test class"""
        self.setup_test_data()

    def setup_test_data(self):
        """Setup test data and directories"""
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        # Create sample test data if processed data doesn't exist
        self.ensure_test_data_exists()

    def ensure_test_data_exists(self):
        """Ensure test data exists by running Task 1 if needed"""
        processed_dir = Path("data/processed")
        features_path = processed_dir / "features.csv"
        target_path = processed_dir / "target.csv"

        if not features_path.exists() or not target_path.exists():
            print("Processed data not found. Running Task 1 first...")
            try:
                from data_acquisition_eda import main as task1_main

                task1_main()
                print("Task 1 completed. Proceeding with Task 2 tests...")
            except Exception as e:
                # Create minimal test data if Task 1 fails
                print(f"Task 1 failed ({e}). Creating minimal test data...")
                self.create_minimal_test_data()

    def create_minimal_test_data(self):
        """Create minimal test data for testing purposes"""
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Create sample features
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame(
            {
                "age": np.random.randint(30, 80, n_samples),
                "sex": np.random.randint(0, 2, n_samples),
                "cp": np.random.randint(1, 5, n_samples),
                "trestbps": np.random.randint(90, 200, n_samples),
                "chol": np.random.randint(150, 400, n_samples),
                "fbs": np.random.randint(0, 2, n_samples),
                "restecg": np.random.randint(0, 3, n_samples),
                "thalach": np.random.randint(80, 200, n_samples),
                "exang": np.random.randint(0, 2, n_samples),
                "oldpeak": np.random.uniform(0, 5, n_samples),
                "slope": np.random.randint(1, 4, n_samples),
                "ca": np.random.randint(0, 4, n_samples),
                "thal": np.random.randint(3, 8, n_samples),
            }
        )

        # Add some missing values
        X.loc[0:2, "ca"] = np.nan
        X.loc[0:1, "thal"] = np.nan

        # Create sample target
        y = pd.DataFrame({"num": np.random.randint(0, 5, n_samples)})

        # Save test data
        X.to_csv(processed_dir / "features.csv", index=False)
        y.to_csv(processed_dir / "target.csv", index=False)

        print("Minimal test data created successfully")

    def test_load_processed_data(self):
        """Test loading processed data"""
        try:
            X, y = load_processed_data()

            assert isinstance(X, pd.DataFrame), "Features should be DataFrame"
            assert isinstance(y, pd.DataFrame), "Target should be DataFrame"
            assert X.shape[0] > 0, "Should have samples"
            assert X.shape[1] > 0, "Should have features"
            assert X.shape[0] == y.shape[0], "Features and target should have same length"

            print(f"Data loading test passed: {X.shape}")

        except Exception as e:
            raise Exception(f"Data loading failed: {e}")

    def test_handle_missing_values(self):
        """Test missing value handling"""
        X, _ = load_processed_data()

        # Check if there are missing values to handle
        missing_before = X.isnull().sum().sum()

        X_clean = handle_missing_values(X)

        # Check that missing values are handled
        missing_after = X_clean.isnull().sum().sum()

        assert missing_after == 0, "All missing values should be handled"
        assert X_clean.shape == X.shape, "Shape should remain the same"

        print(f"Missing values handled: {missing_before} → {missing_after}")

    def test_prepare_target_variable(self):
        """Test target variable preparation"""
        _, y = load_processed_data()

        y_binary, y_multiclass, target_info = prepare_target_variable(y)

        # Test binary target
        assert isinstance(y_binary, pd.Series), "Binary target should be Series"
        assert set(y_binary.unique()).issubset({0, 1}), "Binary target should only have 0 and 1"

        # Test multiclass target
        assert isinstance(y_multiclass, pd.Series), "Multiclass target should be Series"

        # Test target info
        assert isinstance(target_info, dict), "Target info should be dictionary"
        assert "binary_distribution" in target_info, "Should have binary distribution"
        assert "multiclass_distribution" in target_info, "Should have multiclass distribution"

        print(f"Target preparation test passed: {len(y_binary.unique())} binary classes")

    def test_engineer_features(self):
        """Test feature engineering"""
        X, _ = load_processed_data()
        X_clean = handle_missing_values(X)

        X_eng = engineer_features(X_clean)

        # Test that new features are added
        assert X_eng.shape[1] > X_clean.shape[1], "Should add new features"
        assert X_eng.shape[0] == X_clean.shape[0], "Should keep same number of samples"

        # Test specific engineered features
        expected_new_features = [
            "age_group",
            "chol_age_ratio",
            "heart_rate_reserve",
            "risk_score",
            "age_sex_interaction",
            "cp_exang_interaction",
        ]

        for feature in expected_new_features:
            assert feature in X_eng.columns, f"Should have engineered feature: {feature}"

        print(f"Feature engineering test passed: {X_clean.shape[1]} → {X_eng.shape[1]} features")

    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline creation"""
        pipeline = create_preprocessing_pipeline()

        # Test that pipeline is created
        assert pipeline is not None, "Pipeline should be created"
        assert hasattr(pipeline, "fit"), "Pipeline should have fit method"
        assert hasattr(pipeline, "transform"), "Pipeline should have transform method"

        print("Preprocessing pipeline test passed")

    def test_data_splitting(self):
        """Test data splitting functionality"""
        X, y = load_processed_data()
        X_clean = handle_missing_values(X)
        y_binary, _, _ = prepare_target_variable(y)

        X_train, X_test, y_train, y_test = split_data(X_clean, y_binary)

        # Test shapes
        assert len(X_train) > len(X_test), "Training set should be larger"
        assert len(X_train) + len(X_test) == len(X_clean), "Should split all data"
        assert len(y_train) == len(X_train), "Training target should match features"
        assert len(y_test) == len(X_test), "Test target should match features"

        # Test stratification (approximately balanced)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.2, "Should maintain class balance"

        print(f"Data splitting test passed: {len(X_train)} train, {len(X_test)} test")

    def test_model_definition(self):
        """Test model definition"""
        models = define_models()

        assert isinstance(models, dict), "Should return dictionary of models"
        assert len(models) > 0, "Should define at least one model"

        # Test that all models have required methods
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} should have fit method"
            assert hasattr(model, "predict"), f"{name} should have predict method"

        expected_models = ["logistic_regression", "random_forest", "gradient_boosting", "svm"]
        for model_name in expected_models:
            assert model_name in models, f"Should have {model_name} model"

        print(f"Model definition test passed: {len(models)} models defined")

    def test_cross_validation(self):
        """Test cross-validation functionality"""
        # Prepare data
        X, y = load_processed_data()
        X_clean = handle_missing_values(X)
        X_eng = engineer_features(X_clean)
        y_binary, _, _ = prepare_target_variable(y)
        X_train, _, y_train, _ = split_data(X_eng, y_binary)

        # Define models (use subset for faster testing)
        models = {
            "logistic_regression": define_models()["logistic_regression"],
            "random_forest": define_models()["random_forest"],
        }

        # Perform cross-validation
        cv_results = perform_cross_validation(models, X_train, y_train, cv_folds=3)

        assert isinstance(cv_results, dict), "Should return dictionary"
        assert len(cv_results) == len(models), "Should have results for all models"

        for name, results in cv_results.items():
            assert "scores" in results, f"{name} should have scores"
            assert "mean" in results, f"{name} should have mean score"
            assert "std" in results, f"{name} should have std score"
            assert results["mean"] > 0, f"{name} should have positive score"
            assert results["mean"] <= 1, f"{name} should have score <= 1"

        print(f"Cross-validation test passed: {len(cv_results)} models evaluated")

    def test_model_evaluation(self):
        """Test model evaluation functionality"""
        # Prepare data
        X, y = load_processed_data()
        X_clean = handle_missing_values(X)
        X_eng = engineer_features(X_clean)
        y_binary, _, _ = prepare_target_variable(y)
        X_train, X_test, y_train, y_test = split_data(X_eng, y_binary)

        # Use simple models for testing
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        test_models = {
            "test_logistic": {
                "model": Pipeline(
                    [("preprocessor", create_preprocessing_pipeline()), ("classifier", LogisticRegression(random_state=42))]
                )
            }
        }

        # Evaluate models
        evaluation_results = evaluate_models(test_models, X_train, X_test, y_train, y_test)

        assert isinstance(evaluation_results, dict), "Should return dictionary"
        assert len(evaluation_results) > 0, "Should have evaluation results"

        for name, results in evaluation_results.items():
            assert "metrics" in results, f"{name} should have metrics"
            assert "y_pred" in results, f"{name} should have predictions"

            metrics = results["metrics"]
            assert "accuracy" in metrics, "Should have accuracy"
            assert "roc_auc" in metrics, "Should have ROC-AUC"
            assert 0 <= metrics["accuracy"] <= 1, "Accuracy should be between 0 and 1"
            assert 0 <= metrics["roc_auc"] <= 1, "ROC-AUC should be between 0 and 1"

        print(f"Model evaluation test passed: {len(evaluation_results)} models evaluated")


def test_full_task2_pipeline():
    """Test the complete Task 2 pipeline"""
    try:
        # Import and run the main function
        from feature_engineering import main

        evaluation_results, X_engineered, y_binary = main()

        # Basic checks
        assert isinstance(evaluation_results, dict), "Should return evaluation results"
        assert isinstance(X_engineered, pd.DataFrame), "Should return engineered features"
        assert isinstance(y_binary, pd.Series), "Should return binary target"

        # Check that models were saved
        models_dir = Path("models")
        expected_files = ["best_model.joblib", "evaluation_results.json", "feature_names.json"]

        for filename in expected_files:
            file_path = models_dir / filename
            assert file_path.exists(), f"Expected file not found: {filename}"

        # Test loading saved model
        best_model = joblib.load(models_dir / "best_model.joblib")
        assert hasattr(best_model, "predict"), "Saved model should have predict method"

        print("Full Task 2 pipeline test passed")

    except Exception as e:
        raise Exception(f"Full pipeline test failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestTask2FeatureEngineering()

    print("Running Task 2 Tests...")
    print("=" * 50)

    try:
        test_instance.test_load_processed_data()
        test_instance.test_handle_missing_values()
        test_instance.test_prepare_target_variable()
        test_instance.test_engineer_features()
        test_instance.test_preprocessing_pipeline()
        test_instance.test_data_splitting()
        test_instance.test_model_definition()
        test_instance.test_cross_validation()
        test_instance.test_model_evaluation()
        test_full_task2_pipeline()

        print("\n" + "=" * 50)
        print("All Task 2 tests passed!")
        print("=" * 50)

    except Exception as e:
        print(f"\nTest failed: {e}")
        raise
