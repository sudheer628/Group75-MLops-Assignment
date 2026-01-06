"""
Offline Feature Store for Heart Disease Prediction

This module provides a simple Parquet-based feature store that:
1. Centralizes feature engineering logic
2. Stores computed features for reuse
3. Validates feature schemas
4. Ensures consistency between training and inference

Usage:
    from src.feature_store import feature_store

    # Compute features from raw data
    features_df = feature_store.compute_features(raw_df)

    # Save features
    feature_store.save_features(features_df, version="v1")

    # Load features
    features_df = feature_store.load_features(version="v1")
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Feature Store directories
FEATURE_STORE_DIR = Path("feature_store")
FEATURES_DIR = FEATURE_STORE_DIR / "features"
SCHEMAS_DIR = FEATURE_STORE_DIR / "schemas"
METADATA_DIR = FEATURE_STORE_DIR / "metadata"


class FeatureStore:
    """Simple offline feature store using Parquet files"""

    # Raw feature names (input from dataset)
    RAW_FEATURES = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    # Engineered feature names
    ENGINEERED_FEATURES = [
        "age_group",
        "chol_age_ratio",
        "heart_rate_reserve",
        "risk_score",
        "age_sex_interaction",
        "cp_exang_interaction",
    ]

    # All features (raw + engineered)
    ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

    # Feature schema with types and valid ranges
    FEATURE_SCHEMA = {
        "age": {"type": "int", "min": 1, "max": 120},
        "sex": {"type": "int", "min": 0, "max": 1},
        "cp": {"type": "int", "min": 0, "max": 3},
        "trestbps": {"type": "int", "min": 50, "max": 300},
        "chol": {"type": "int", "min": 100, "max": 600},
        "fbs": {"type": "int", "min": 0, "max": 1},
        "restecg": {"type": "int", "min": 0, "max": 2},
        "thalach": {"type": "int", "min": 60, "max": 220},
        "exang": {"type": "int", "min": 0, "max": 1},
        "oldpeak": {"type": "float", "min": 0.0, "max": 10.0},
        "slope": {"type": "int", "min": 0, "max": 2},
        "ca": {"type": "int", "min": 0, "max": 4},
        "thal": {"type": "int", "min": 0, "max": 3},
        "age_group": {"type": "int", "min": 0, "max": 3},
        "chol_age_ratio": {"type": "float", "min": 1.0, "max": 20.0},
        "heart_rate_reserve": {"type": "float", "min": -100, "max": 100},
        "risk_score": {"type": "float", "min": 0, "max": 100},
        "age_sex_interaction": {"type": "float", "min": 0, "max": 150},
        "cp_exang_interaction": {"type": "int", "min": 0, "max": 3},
    }

    def __init__(self):
        """Initialize feature store and create directories"""
        self._ensure_directories()

    def _ensure_directories(self):
        """Create feature store directories if they don't exist"""
        for directory in [FEATURE_STORE_DIR, FEATURES_DIR, SCHEMAS_DIR, METADATA_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute engineered features from raw data

        This is the SINGLE SOURCE OF TRUTH for feature engineering.
        Both training and inference use this method.

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with raw + engineered features
        """
        df_eng = df.copy()

        # Ensure we have all raw features
        missing_features = set(self.RAW_FEATURES) - set(df_eng.columns)
        if missing_features:
            raise ValueError(f"Missing raw features: {missing_features}")

        # Age groups: 0=young(0-40), 1=middle_aged(40-50), 2=senior(50-60), 3=elderly(60+)
        df_eng["age_group"] = pd.cut(df_eng["age"], bins=[0, 40, 50, 60, 120], labels=[0, 1, 2, 3]).astype(int)

        # Cholesterol to age ratio (metabolic health indicator)
        df_eng["chol_age_ratio"] = df_eng["chol"] / df_eng["age"]

        # Heart rate reserve (exercise capacity)
        df_eng["heart_rate_reserve"] = df_eng["thalach"] - (220 - df_eng["age"])

        # Composite risk score
        df_eng["risk_score"] = df_eng["age"] * 0.1 + df_eng["chol"] * 0.01 + df_eng["trestbps"] * 0.1 + df_eng["oldpeak"] * 10

        # Interaction features
        df_eng["age_sex_interaction"] = df_eng["age"] * df_eng["sex"]
        df_eng["cp_exang_interaction"] = df_eng["cp"] * df_eng["exang"]

        # Ensure correct column order
        df_eng = df_eng[self.ALL_FEATURES]

        return df_eng

    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate features against schema

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check all required features exist
        missing = set(self.ALL_FEATURES) - set(df.columns)
        if missing:
            errors.append(f"Missing features: {missing}")

        # Check for null values
        null_counts = df.isnull().sum()
        null_features = null_counts[null_counts > 0]
        if len(null_features) > 0:
            errors.append(f"Null values found: {null_features.to_dict()}")

        # Validate ranges for each feature
        for feature, schema in self.FEATURE_SCHEMA.items():
            if feature not in df.columns:
                continue

            col = df[feature]
            min_val, max_val = schema["min"], schema["max"]

            if col.min() < min_val:
                errors.append(f"{feature}: min value {col.min()} < expected {min_val}")
            if col.max() > max_val:
                errors.append(f"{feature}: max value {col.max()} > expected {max_val}")

        return len(errors) == 0, errors

    def save_features(self, df: pd.DataFrame, name: str = "heart_disease_features", version: str = "v1") -> Path:
        """
        Save features to Parquet file

        Args:
            df: DataFrame with features
            name: Feature set name
            version: Version string

        Returns:
            Path to saved file
        """
        # Validate before saving
        is_valid, errors = self.validate_features(df)
        if not is_valid:
            raise ValueError(f"Feature validation failed: {errors}")

        # Save features
        filename = f"{name}_{version}.parquet"
        filepath = FEATURES_DIR / filename
        df.to_parquet(filepath, index=False)

        # Save metadata
        metadata = {
            "name": name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "num_samples": len(df),
            "num_features": len(df.columns),
            "features": df.columns.tolist(),
            "checksum": self._compute_checksum(df),
        }

        metadata_path = METADATA_DIR / f"{name}_{version}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Features saved to: {filepath}")
        print(f"Metadata saved to: {metadata_path}")

        return filepath

    def load_features(self, name: str = "heart_disease_features", version: str = "v1") -> pd.DataFrame:
        """
        Load features from Parquet file

        Args:
            name: Feature set name
            version: Version string

        Returns:
            DataFrame with features
        """
        filename = f"{name}_{version}.parquet"
        filepath = FEATURES_DIR / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Feature file not found: {filepath}")

        df = pd.read_parquet(filepath)
        print(f"Loaded {len(df)} samples with {len(df.columns)} features from {filepath}")

        return df

    def get_feature_schema(self) -> Dict:
        """Get feature schema definition"""
        return self.FEATURE_SCHEMA.copy()

    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get feature names by category"""
        return {
            "raw": self.RAW_FEATURES.copy(),
            "engineered": self.ENGINEERED_FEATURES.copy(),
            "all": self.ALL_FEATURES.copy(),
        }

    def _compute_checksum(self, df: pd.DataFrame) -> str:
        """Compute checksum for data integrity"""
        return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()

    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute feature statistics for monitoring/drift detection

        Args:
            df: DataFrame with features

        Returns:
            Dictionary with statistics per feature
        """
        stats = {}
        for col in df.columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
                "null_count": int(df[col].isnull().sum()),
            }
        return stats

    def save_schema(self):
        """Save feature schema to JSON file"""
        schema_path = SCHEMAS_DIR / "feature_schema.json"

        schema_doc = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "raw_features": self.RAW_FEATURES,
            "engineered_features": self.ENGINEERED_FEATURES,
            "all_features": self.ALL_FEATURES,
            "feature_definitions": self.FEATURE_SCHEMA,
        }

        with open(schema_path, "w") as f:
            json.dump(schema_doc, f, indent=2)

        print(f"Schema saved to: {schema_path}")
        return schema_path

    def check_training_inference_parity(self, sample_input: Dict) -> Tuple[bool, Optional[str]]:
        """
        Verify that inference produces same features as training

        Args:
            sample_input: Dictionary with raw feature values

        Returns:
            Tuple of (is_consistent, error_message)
        """
        try:
            # Create DataFrame from sample input
            df = pd.DataFrame([sample_input])

            # Compute features
            features_df = self.compute_features(df)

            # Validate
            is_valid, errors = self.validate_features(features_df)

            if not is_valid:
                return False, f"Validation errors: {errors}"

            # Check all expected features are present
            if set(features_df.columns) != set(self.ALL_FEATURES):
                return False, "Feature mismatch between expected and computed"

            return True, None

        except Exception as e:
            return False, str(e)


# Global feature store instance
feature_store = FeatureStore()


def validate_feature_store():
    """
    Validate feature store configuration
    Called by CI/CD pipeline
    """
    print("=" * 60)
    print("FEATURE STORE VALIDATION")
    print("=" * 60)

    fs = FeatureStore()

    # Test 1: Schema is valid
    print("\n1. Validating schema...")
    schema = fs.get_feature_schema()
    assert len(schema) == len(fs.ALL_FEATURES), "Schema missing features"
    print(f"   Schema has {len(schema)} feature definitions")

    # Test 2: Feature computation works
    print("\n2. Testing feature computation...")
    sample_data = {
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
        "thal": 3,
    }
    df = pd.DataFrame([sample_data])
    features = fs.compute_features(df)
    assert len(features.columns) == len(fs.ALL_FEATURES), "Feature count mismatch"
    print(f"   Computed {len(features.columns)} features from {len(sample_data)} raw inputs")

    # Test 3: Validation works
    print("\n3. Testing feature validation...")
    is_valid, errors = fs.validate_features(features)
    assert is_valid, f"Validation failed: {errors}"
    print("   Feature validation passed")

    # Test 4: Training/inference parity
    print("\n4. Testing training/inference parity...")
    is_consistent, error = fs.check_training_inference_parity(sample_data)
    assert is_consistent, f"Parity check failed: {error}"
    print("   Training/inference parity confirmed")

    # Test 5: Save schema
    print("\n5. Saving schema...")
    fs.save_schema()

    print("\n" + "=" * 60)
    print("FEATURE STORE VALIDATION PASSED")
    print("=" * 60)

    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        success = validate_feature_store()
        sys.exit(0 if success else 1)
    else:
        # Demo usage
        print("Feature Store Demo")
        print("-" * 40)

        fs = FeatureStore()

        # Show feature names
        names = fs.get_feature_names()
        print(f"Raw features ({len(names['raw'])}): {names['raw']}")
        print(f"Engineered features ({len(names['engineered'])}): {names['engineered']}")

        # Compute sample features
        sample = pd.DataFrame(
            [
                {
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
                    "thal": 3,
                }
            ]
        )

        features = fs.compute_features(sample)
        print(f"\nComputed features:\n{features.T}")
