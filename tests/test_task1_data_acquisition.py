"""
Unit tests for Task 1: Data Acquisition & EDA
Tests the data loading, quality checks, and EDA functionality
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_acquisition_eda import (
    check_data_quality,
    explore_dataset_info,
    load_heart_disease_data,
    perform_eda_analysis,
    save_processed_data,
)


class TestTask1DataAcquisition:
    """Test class for Task 1 functionality"""

    def __init__(self):
        """Initialize test class"""
        self.setup_directories()

    def setup_directories(self):
        """Setup test directories"""
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"

        # Create directories if they don't exist
        for directory in [self.data_dir, self.processed_dir, self.raw_dir]:
            directory.mkdir(exist_ok=True)

    def test_load_heart_disease_data(self):
        """Test dataset loading functionality"""
        try:
            X, y, metadata = load_heart_disease_data()

            # Test data shapes
            assert isinstance(X, pd.DataFrame), "Features should be a DataFrame"
            assert isinstance(y, pd.DataFrame), "Target should be a DataFrame"
            assert X.shape[0] > 0, "Features should have rows"
            assert y.shape[0] > 0, "Target should have rows"
            assert X.shape[0] == y.shape[0], "Features and target should have same number of rows"

            # Test expected number of features (should be around 13 for heart disease dataset)
            assert X.shape[1] >= 10, f"Expected at least 10 features, got {X.shape[1]}"

            # Test metadata
            assert isinstance(metadata, dict), "Metadata should be a dictionary"

            print(f"Dataset loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")

        except Exception as e:
            raise Exception(f"Failed to load dataset: {e}")

    def test_data_quality_check(self):
        """Test data quality assessment"""
        # Load data first
        X, y, _ = load_heart_disease_data()

        # Run quality check
        quality_report = check_data_quality(X, y)

        # Test quality report structure
        assert isinstance(quality_report, dict), "Quality report should be a dictionary"
        assert "missing_values" in quality_report, "Should report missing values"
        assert "duplicates" in quality_report, "Should report duplicates"
        assert "dtypes" in quality_report, "Should report data types"

        # Test missing values report
        missing_values = quality_report["missing_values"]
        assert isinstance(missing_values, dict), "Missing values should be a dictionary"

        # Test duplicates
        duplicates = quality_report["duplicates"]
        assert isinstance(duplicates, (int, np.integer)), "Duplicates should be an integer"
        assert duplicates >= 0, "Duplicates count should be non-negative"

        print(f"Data quality check passed: {duplicates} duplicates found")

    def test_eda_analysis(self):
        """Test EDA analysis functionality"""
        # Load data
        X, y, _ = load_heart_disease_data()

        # This should run without errors
        try:
            perform_eda_analysis(X, y)
            print("EDA analysis completed successfully")
        except Exception as e:
            raise Exception(f"EDA analysis failed: {e}")

    def test_save_processed_data(self):
        """Test data saving functionality"""
        # Load data
        X, y, _ = load_heart_disease_data()

        # Create a simple quality report
        quality_report = {"missing_values": X.isnull().sum().to_dict(), "duplicates": 0, "dtypes": X.dtypes.to_dict()}

        # Save data
        save_processed_data(X, y, quality_report)

        # Check if files were created
        expected_files = [
            self.raw_dir / "heart_disease_raw.csv",
            self.processed_dir / "features.csv",
            self.processed_dir / "target.csv",
            self.processed_dir / "data_quality_report.json",
        ]

        for file_path in expected_files:
            assert file_path.exists(), f"Expected file not created: {file_path}"

        # Test loading saved data
        saved_X = pd.read_csv(self.processed_dir / "features.csv")
        saved_y = pd.read_csv(self.processed_dir / "target.csv")

        assert saved_X.shape == X.shape, "Saved features shape mismatch"
        assert saved_y.shape == y.shape, "Saved target shape mismatch"

        # Test quality report JSON
        with open(self.processed_dir / "data_quality_report.json", "r") as f:
            saved_report = json.load(f)

        assert isinstance(saved_report, dict), "Saved quality report should be a dictionary"

        print("Data saving and loading test passed")

    def test_data_consistency(self):
        """Test data consistency and expected ranges"""
        X, y, _ = load_heart_disease_data()

        # Test feature value ranges (basic sanity checks)
        if "age" in X.columns:
            assert X["age"].min() >= 0, "Age should be non-negative"
            assert X["age"].max() <= 120, "Age should be reasonable"

        if "sex" in X.columns:
            unique_sex = X["sex"].unique()
            assert len(unique_sex) <= 3, "Sex should have at most 3 unique values"

        # Test target values
        target_col = y.columns[0]
        target_values = y[target_col].unique()
        assert len(target_values) >= 2, "Should have at least 2 target classes"

        print(f"Data consistency check passed: {len(target_values)} target classes")


def test_full_task1_pipeline():
    """Test the complete Task 1 pipeline"""
    try:
        # Import and run the main function
        from data_acquisition_eda import main

        X, y, quality_report = main()

        # Basic checks
        assert isinstance(X, pd.DataFrame), "Pipeline should return DataFrame for features"
        assert isinstance(y, pd.DataFrame), "Pipeline should return DataFrame for target"
        assert isinstance(quality_report, dict), "Pipeline should return quality report"

        print("Full Task 1 pipeline test passed")

    except Exception as e:
        raise Exception(f"Full pipeline test failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestTask1DataAcquisition()

    print("Running Task 1 Tests...")
    print("=" * 50)

    try:
        test_instance.test_load_heart_disease_data()
        test_instance.test_data_quality_check()
        test_instance.test_eda_analysis()
        test_instance.test_save_processed_data()
        test_instance.test_data_consistency()
        test_full_task1_pipeline()

        print("\n" + "=" * 50)
        print("All Task 1 tests passed!")
        print("=" * 50)

    except Exception as e:
        print(f"\nTest failed: {e}")
        raise
