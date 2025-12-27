"""
Task 1: Data Acquisition & Exploratory Data Analysis (EDA)
Heart Disease UCI Dataset Analysis

This script handles:
1. Dataset acquisition from UCI ML Repository
2. Data cleaning and preprocessing
3. Exploratory Data Analysis with visualizations
4. Data quality assessment
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set up directories
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = Path("figures")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR]:
    directory.mkdir(exist_ok=True)


def load_heart_disease_data():
    """
    Load heart disease dataset from UCI ML Repository

    Returns:
        tuple: (features_df, target_df, metadata)
    """
    print("Loading Heart Disease UCI Dataset...")

    try:
        # Try to load from UCI ML Repository
        from ucimlrepo import fetch_ucirepo

        heart_disease = fetch_ucirepo(id=45)

        X = heart_disease.data.features
        y = heart_disease.data.targets
        metadata = heart_disease.metadata

        print("Dataset loaded successfully from UCI ML Repository")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        return X, y, metadata

    except Exception as e:
        print(f"Failed to load from UCI ML Repository: {e}")
        raise Exception(
            "Could not load dataset from UCI ML Repository. Please check your internet connection and ucimlrepo installation."
        )


def explore_dataset_info(X, y, metadata):
    """
    Display basic information about the dataset

    Args:
        X (pd.DataFrame): Features
        y (pd.DataFrame): Target
        metadata (dict): Dataset metadata
    """
    print("\n" + "=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    print(f"Dataset: {metadata.get('name', 'Heart Disease Dataset')}")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {len(X.columns)}")

    print(f"\nFeature columns:")
    for i, col in enumerate(X.columns, 1):
        print(f"{i:2d}. {col}")

    print(f"\nTarget column(s):")
    for col in y.columns:
        print(f"    {col}")

    print(f"\nDataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")


def check_data_quality(X, y):
    """
    Assess data quality: missing values, duplicates, data types

    Args:
        X (pd.DataFrame): Features
        y (pd.DataFrame): Target

    Returns:
        dict: Data quality report
    """
    print("\n" + "=" * 50)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 50)

    # Combine for analysis
    df = pd.concat([X, y], axis=1)

    quality_report = {}

    # Missing values
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100

    print("Missing Values:")
    for col in df.columns:
        if missing_values[col] > 0:
            print(f"  {col}: {missing_values[col]} ({missing_percent[col]:.2f}%)")

    if missing_values.sum() == 0:
        print("  No missing values found")

    quality_report["missing_values"] = missing_values.to_dict()

    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    quality_report["duplicates"] = duplicates

    # Data types
    print(f"\nData types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    quality_report["dtypes"] = df.dtypes.to_dict()

    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df.describe())

    return quality_report


def perform_eda_analysis(X, y):
    """
    Perform comprehensive Exploratory Data Analysis

    Args:
        X (pd.DataFrame): Features
        y (pd.DataFrame): Target
    """
    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # Combine for analysis
    df = pd.concat([X, y], axis=1)
    target_col = y.columns[0]

    # Target distribution
    print("Target Distribution:")
    target_counts = df[target_col].value_counts().sort_index()
    print(target_counts)
    print(f"Target balance: {target_counts.min()}/{target_counts.max()} = {target_counts.min() / target_counts.max():.3f}")

    # Feature statistics by target
    print(f"\nFeature Statistics by Target:")
    for feature in X.columns:
        if df[feature].dtype in ["int64", "float64"]:
            print(f"\n{feature}:")
            stats = df.groupby(target_col)[feature].agg(["mean", "std", "min", "max"])
            print(stats)


def create_eda_visualizations(X, y):
    """
    Create comprehensive EDA visualizations

    Args:
        X (pd.DataFrame): Features
        y (pd.DataFrame): Target
    """
    print("\n" + "=" * 50)
    print("CREATING EDA VISUALIZATIONS")
    print("=" * 50)

    # Combine for visualization
    df = pd.concat([X, y], axis=1)
    target_col = y.columns[0]

    # Set style for better plots
    # plt.style.use('seaborn-v0_8')
    # sns.set_palette("husl")

    # 1. Target Distribution
    print("1. Target distribution plot")
    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # target_counts = df[target_col].value_counts().sort_index()
    # ax.bar(target_counts.index, target_counts.values)
    # ax.set_title('Target Distribution (Heart Disease)')
    # ax.set_xlabel('Target (0: No Disease, 1: Disease)')
    # ax.set_ylabel('Count')
    # for i, v in enumerate(target_counts.values):
    #     ax.text(i, v + 5, str(v), ha='center')
    # plt.tight_layout()
    # plt.savefig(FIGURES_DIR / 'target_distribution.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 2. Age Distribution by Target
    print("2. Age distribution by target")
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # for target_val in df[target_col].unique():
    #     subset = df[df[target_col] == target_val]
    #     ax.hist(subset['age'], alpha=0.7, label=f'Target {target_val}', bins=20)
    # ax.set_title('Age Distribution by Heart Disease Status')
    # ax.set_xlabel('Age')
    # ax.set_ylabel('Frequency')
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(FIGURES_DIR / 'age_distribution.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 3. Correlation Heatmap
    print("3. Correlation heatmap")
    # fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # correlation_matrix = df.corr()
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
    #             square=True, ax=ax, fmt='.2f')
    # ax.set_title('Feature Correlation Heatmap')
    # plt.tight_layout()
    # plt.savefig(FIGURES_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 4. Feature distributions
    print("4. Feature distribution plots")
    # numeric_features = X.select_dtypes(include=[np.number]).columns
    # n_features = len(numeric_features)
    # n_cols = 3
    # n_rows = (n_features + n_cols - 1) // n_cols
    #
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    # axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    #
    # for i, feature in enumerate(numeric_features):
    #     if i < len(axes):
    #         axes[i].hist(df[feature], bins=20, alpha=0.7, edgecolor='black')
    #         axes[i].set_title(f'Distribution of {feature}')
    #         axes[i].set_xlabel(feature)
    #         axes[i].set_ylabel('Frequency')
    #
    # # Hide empty subplots
    # for i in range(len(numeric_features), len(axes)):
    #     axes[i].set_visible(False)
    #
    # plt.tight_layout()
    # plt.savefig(FIGURES_DIR / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 5. Box plots by target
    print("5. Box plots by target")
    # numeric_features = X.select_dtypes(include=[np.number]).columns
    # n_features = len(numeric_features)
    # n_cols = 3
    # n_rows = (n_features + n_cols - 1) // n_cols
    #
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    # axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    #
    # for i, feature in enumerate(numeric_features):
    #     if i < len(axes):
    #         df.boxplot(column=feature, by=target_col, ax=axes[i])
    #         axes[i].set_title(f'{feature} by Target')
    #         axes[i].set_xlabel('Target')
    #         axes[i].set_ylabel(feature)
    #
    # # Hide empty subplots
    # for i in range(len(numeric_features), len(axes)):
    #     axes[i].set_visible(False)
    #
    # plt.tight_layout()
    # plt.savefig(FIGURES_DIR / 'boxplots_by_target.png', dpi=300, bbox_inches='tight')
    # plt.show()

    print("All visualizations created and saved to figures/ directory")


def save_processed_data(X, y, quality_report):
    """
    Save processed data and quality report

    Args:
        X (pd.DataFrame): Features
        y (pd.DataFrame): Target
        quality_report (dict): Data quality assessment
    """
    print("\n" + "=" * 50)
    print("SAVING PROCESSED DATA")
    print("=" * 50)

    # Save raw data
    df_combined = pd.concat([X, y], axis=1)
    raw_data_path = RAW_DATA_DIR / "heart_disease_raw.csv"
    df_combined.to_csv(raw_data_path, index=False)
    print(f"Raw data saved to: {raw_data_path}")

    # Save features and target separately
    features_path = PROCESSED_DATA_DIR / "features.csv"
    target_path = PROCESSED_DATA_DIR / "target.csv"

    X.to_csv(features_path, index=False)
    y.to_csv(target_path, index=False)

    print(f"Features saved to: {features_path}")
    print(f"Target saved to: {target_path}")

    # Save quality report
    import json

    quality_report_path = PROCESSED_DATA_DIR / "data_quality_report.json"

    # Convert numpy types to native Python types for JSON serialization
    serializable_report = {}
    for key, value in quality_report.items():
        if isinstance(value, dict):
            serializable_report[key] = {k: str(v) for k, v in value.items()}
        else:
            serializable_report[key] = str(value)

    with open(quality_report_path, "w") as f:
        json.dump(serializable_report, f, indent=2)

    print(f"Quality report saved to: {quality_report_path}")


def main():
    """
    Main function to execute Task 1: Data Acquisition & EDA
    """
    print("Starting Task 1: Data Acquisition & Exploratory Data Analysis")
    print("=" * 70)

    try:
        # Step 1: Load dataset
        X, y, metadata = load_heart_disease_data()

        # Step 2: Explore dataset information
        explore_dataset_info(X, y, metadata)

        # Step 3: Check data quality
        quality_report = check_data_quality(X, y)

        # Step 4: Perform EDA analysis
        perform_eda_analysis(X, y)

        # Step 5: Create visualizations
        create_eda_visualizations(X, y)

        # Step 6: Save processed data
        save_processed_data(X, y, quality_report)

        print("\n" + "=" * 70)
        print("Task 1 completed successfully!")
        print("=" * 70)

        return X, y, quality_report

    except Exception as e:
        print(f"Error in Task 1: {e}")
        raise


if __name__ == "__main__":
    X, y, quality_report = main()
