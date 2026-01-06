"""
Tests for the Feature Store

These tests verify that:
1. Feature computation works correctly
2. Feature validation catches invalid data
3. Training/inference parity is maintained
4. Schema definitions are correct
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_store import FeatureStore, feature_store


class TestFeatureStore:
    """Test suite for FeatureStore class"""
    
    @pytest.fixture
    def sample_raw_data(self):
        """Sample raw input data"""
        return pd.DataFrame([{
            'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 140,
            'chol': 250, 'fbs': 0, 'restecg': 1, 'thalach': 150,
            'exang': 0, 'oldpeak': 1.5, 'slope': 2, 'ca': 0, 'thal': 3
        }])
    
    @pytest.fixture
    def sample_raw_data_multiple(self):
        """Multiple rows of sample data"""
        return pd.DataFrame([
            {'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 140, 'chol': 250, 
             'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0, 
             'oldpeak': 1.5, 'slope': 2, 'ca': 0, 'thal': 3},
            {'age': 35, 'sex': 0, 'cp': 0, 'trestbps': 120, 'chol': 200, 
             'fbs': 0, 'restecg': 0, 'thalach': 180, 'exang': 0, 
             'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 2},
            {'age': 70, 'sex': 1, 'cp': 2, 'trestbps': 160, 'chol': 300, 
             'fbs': 1, 'restecg': 2, 'thalach': 120, 'exang': 1, 
             'oldpeak': 3.0, 'slope': 0, 'ca': 2, 'thal': 1},
        ])
    
    def test_feature_store_initialization(self):
        """Test that feature store initializes correctly"""
        fs = FeatureStore()
        assert fs is not None
        assert len(fs.RAW_FEATURES) == 13
        assert len(fs.ENGINEERED_FEATURES) == 6
        assert len(fs.ALL_FEATURES) == 19
    
    def test_compute_features_single_row(self, sample_raw_data):
        """Test feature computation for single row"""
        fs = FeatureStore()
        features = fs.compute_features(sample_raw_data)
        
        # Check all features are present
        assert len(features.columns) == 19
        assert set(features.columns) == set(fs.ALL_FEATURES)
        
        # Check engineered features are computed
        assert 'age_group' in features.columns
        assert 'chol_age_ratio' in features.columns
        assert 'heart_rate_reserve' in features.columns
        assert 'risk_score' in features.columns
    
    def test_compute_features_multiple_rows(self, sample_raw_data_multiple):
        """Test feature computation for multiple rows"""
        fs = FeatureStore()
        features = fs.compute_features(sample_raw_data_multiple)
        
        assert len(features) == 3
        assert len(features.columns) == 19
    
    def test_compute_features_age_groups(self):
        """Test age group computation"""
        fs = FeatureStore()
        
        # Test different age ranges
        test_cases = [
            (30, 0),   # young (0-40)
            (45, 1),   # middle_aged (40-50)
            (55, 2),   # senior (50-60)
            (70, 3),   # elderly (60+)
        ]
        
        for age, expected_group in test_cases:
            data = pd.DataFrame([{
                'age': age, 'sex': 1, 'cp': 0, 'trestbps': 120,
                'chol': 200, 'fbs': 0, 'restecg': 0, 'thalach': 150,
                'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 2
            }])
            features = fs.compute_features(data)
            assert features['age_group'].iloc[0] == expected_group, \
                f"Age {age} should be group {expected_group}"
    
    def test_compute_features_missing_column(self):
        """Test that missing columns raise error"""
        fs = FeatureStore()
        incomplete_data = pd.DataFrame([{'age': 55, 'sex': 1}])
        
        with pytest.raises(ValueError, match="Missing raw features"):
            fs.compute_features(incomplete_data)
    
    def test_validate_features_valid_data(self, sample_raw_data):
        """Test validation passes for valid data"""
        fs = FeatureStore()
        features = fs.compute_features(sample_raw_data)
        
        is_valid, errors = fs.validate_features(features)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_features_missing_columns(self):
        """Test validation fails for missing columns"""
        fs = FeatureStore()
        incomplete_features = pd.DataFrame([{'age': 55, 'sex': 1}])
        
        is_valid, errors = fs.validate_features(incomplete_features)
        assert is_valid is False
        assert any('Missing features' in e for e in errors)
    
    def test_validate_features_null_values(self, sample_raw_data):
        """Test validation catches null values"""
        fs = FeatureStore()
        features = fs.compute_features(sample_raw_data)
        features.loc[0, 'age'] = None
        
        is_valid, errors = fs.validate_features(features)
        assert is_valid is False
        assert any('Null values' in e for e in errors)
    
    def test_get_feature_schema(self):
        """Test schema retrieval"""
        fs = FeatureStore()
        schema = fs.get_feature_schema()
        
        assert len(schema) == 19
        assert 'age' in schema
        assert schema['age']['type'] == 'int'
        assert schema['age']['min'] == 1
        assert schema['age']['max'] == 120
    
    def test_get_feature_names(self):
        """Test feature names retrieval"""
        fs = FeatureStore()
        names = fs.get_feature_names()
        
        assert 'raw' in names
        assert 'engineered' in names
        assert 'all' in names
        assert len(names['raw']) == 13
        assert len(names['engineered']) == 6
        assert len(names['all']) == 19
    
    def test_training_inference_parity(self):
        """Test that training and inference produce same features"""
        fs = FeatureStore()
        
        sample_input = {
            'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 140,
            'chol': 250, 'fbs': 0, 'restecg': 1, 'thalach': 150,
            'exang': 0, 'oldpeak': 1.5, 'slope': 2, 'ca': 0, 'thal': 3
        }
        
        is_consistent, error = fs.check_training_inference_parity(sample_input)
        assert is_consistent is True
        assert error is None
    
    def test_feature_statistics(self, sample_raw_data_multiple):
        """Test feature statistics computation"""
        fs = FeatureStore()
        features = fs.compute_features(sample_raw_data_multiple)
        
        stats = fs.get_feature_statistics(features)
        
        assert 'age' in stats
        assert 'mean' in stats['age']
        assert 'std' in stats['age']
        assert 'min' in stats['age']
        assert 'max' in stats['age']
    
    def test_global_feature_store_instance(self):
        """Test that global instance works"""
        assert feature_store is not None
        assert isinstance(feature_store, FeatureStore)


class TestFeatureStoreIntegration:
    """Integration tests for feature store"""
    
    def test_end_to_end_workflow(self, tmp_path):
        """Test complete workflow: compute -> validate -> save -> load"""
        # Skip if pyarrow not installed
        pytest.importorskip("pyarrow", reason="pyarrow required for parquet operations")
        
        # Create a temporary feature store
        import src.feature_store as fs_module
        
        # Override directories for testing
        original_features_dir = fs_module.FEATURES_DIR
        original_metadata_dir = fs_module.METADATA_DIR
        
        try:
            fs_module.FEATURES_DIR = tmp_path / "features"
            fs_module.METADATA_DIR = tmp_path / "metadata"
            fs_module.FEATURES_DIR.mkdir(parents=True)
            fs_module.METADATA_DIR.mkdir(parents=True)
            
            fs = FeatureStore()
            
            # Create sample data
            raw_data = pd.DataFrame([
                {'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 140, 'chol': 250, 
                 'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0, 
                 'oldpeak': 1.5, 'slope': 2, 'ca': 0, 'thal': 3},
                {'age': 45, 'sex': 0, 'cp': 1, 'trestbps': 130, 'chol': 220, 
                 'fbs': 0, 'restecg': 0, 'thalach': 160, 'exang': 0, 
                 'oldpeak': 0.5, 'slope': 1, 'ca': 0, 'thal': 2},
            ])
            
            # Compute features
            features = fs.compute_features(raw_data)
            assert len(features) == 2
            
            # Validate
            is_valid, errors = fs.validate_features(features)
            assert is_valid is True
            
            # Save
            filepath = fs.save_features(features, name="test_features", version="v1")
            assert filepath.exists()
            
            # Load
            loaded_features = fs.load_features(name="test_features", version="v1")
            assert len(loaded_features) == 2
            
            # Verify data integrity
            pd.testing.assert_frame_equal(features, loaded_features)
            
        finally:
            # Restore original directories
            fs_module.FEATURES_DIR = original_features_dir
            fs_module.METADATA_DIR = original_metadata_dir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
