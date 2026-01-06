# Feature Store

This is a simple Parquet-based offline feature store for the heart disease prediction model.

## Why we built this

During the project, we noticed that feature engineering logic was duplicated in multiple places:

- `src/feature_engineering.py` (training)
- `app/prediction.py` (inference)

This is a common problem in ML projects - training/serving skew. If someone updates the feature logic in one place but forgets the other, the model will behave differently in production than it did during training.

## How it works

The feature store (`src/feature_store.py`) provides a single source of truth for:

1. Feature definitions - what features exist and their schemas
2. Feature computation - the actual engineering logic
3. Feature validation - checking data quality
4. Feature storage - saving/loading computed features as Parquet files

## Directory structure

```
feature_store/
  features/       - Parquet files with computed features
  schemas/        - JSON schema definitions
  metadata/       - Metadata about saved feature sets
  README.md       - This file
```

## Usage

```python
from src.feature_store import feature_store

# Compute features from raw data
features_df = feature_store.compute_features(raw_df)

# Validate features
is_valid, errors = feature_store.validate_features(features_df)

# Save features
feature_store.save_features(features_df, version="v1")

# Load features
features_df = feature_store.load_features(version="v1")
```

## CI/CD Integration

The feature store validation runs as part of the CI pipeline:

```bash
python src/feature_store.py --validate
```

This checks:

- Schema is valid
- Feature computation works
- Validation logic works
- Training/inference parity is maintained
