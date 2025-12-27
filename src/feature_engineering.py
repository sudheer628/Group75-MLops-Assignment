"""
Task 2: Feature Engineering & Model Development
Heart Disease Prediction Model Training

This script handles:
1. Feature preprocessing and engineering
2. Multiple model training (Logistic Regression, Random Forest, etc.)
3. Cross-validation and hyperparameter tuning
4. Model evaluation and comparison
5. Performance metrics calculation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set up directories
DATA_DIR = Path("data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
FIGURES_DIR = Path("figures")

# Create directories if they don't exist
for directory in [MODELS_DIR, FIGURES_DIR]:
    directory.mkdir(exist_ok=True)

def load_processed_data():
    """
    Load the processed data from Task 1
    
    Returns:
        tuple: (X, y) - features and target
    """
    print("Loading processed data from Task 1...")
    
    features_path = PROCESSED_DATA_DIR / "features.csv"
    target_path = PROCESSED_DATA_DIR / "target.csv"
    
    if not features_path.exists() or not target_path.exists():
        raise FileNotFoundError("Processed data not found. Please run Task 1 first.")
    
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path)
    
    print(f"Features loaded: {X.shape}")
    print(f"Target loaded: {y.shape}")
    
    return X, y

def handle_missing_values(X):
    """
    Handle missing values in the dataset
    
    Args:
        X (pd.DataFrame): Feature matrix
        
    Returns:
        pd.DataFrame: Cleaned feature matrix
    """
    print("\n" + "="*50)
    print("HANDLING MISSING VALUES")
    print("="*50)
    
    # Check for missing values
    missing_before = X.isnull().sum()
    print("Missing values before cleaning:")
    for col in X.columns:
        if missing_before[col] > 0:
            print(f"  {col}: {missing_before[col]}")
    
    X_clean = X.copy()
    
    # Handle missing values based on feature type
    for col in X.columns:
        if X_clean[col].isnull().sum() > 0:
            if X_clean[col].dtype in ['int64', 'float64']:
                # For numeric features, use median imputation
                median_val = X_clean[col].median()
                X_clean[col].fillna(median_val, inplace=True)
                print(f"  Filled {col} missing values with median: {median_val}")
            else:
                # For categorical features, use mode imputation
                mode_val = X_clean[col].mode()[0]
                X_clean[col].fillna(mode_val, inplace=True)
                print(f"  Filled {col} missing values with mode: {mode_val}")
    
    missing_after = X_clean.isnull().sum()
    print(f"\nMissing values after cleaning: {missing_after.sum()}")
    
    return X_clean

def prepare_target_variable(y):
    """
    Prepare target variable for binary classification
    
    Args:
        y (pd.DataFrame): Target variable
        
    Returns:
        tuple: (y_binary, y_multiclass, target_info)
    """
    print("\n" + "="*50)
    print("PREPARING TARGET VARIABLE")
    print("="*50)
    
    target_col = y.columns[0]
    y_original = y[target_col].copy()
    
    print("Original target distribution:")
    print(y_original.value_counts().sort_index())
    
    # Create binary target (0: no disease, 1: disease)
    y_binary = (y_original > 0).astype(int)
    
    print(f"\nBinary target distribution:")
    print(y_binary.value_counts().sort_index())
    
    # Keep multiclass for comparison
    y_multiclass = y_original.copy()
    
    target_info = {
        'original_classes': y_original.unique().tolist(),
        'binary_distribution': y_binary.value_counts().to_dict(),
        'multiclass_distribution': y_multiclass.value_counts().to_dict()
    }
    
    return y_binary, y_multiclass, target_info

def engineer_features(X):
    """
    Create additional engineered features
    
    Args:
        X (pd.DataFrame): Feature matrix
        
    Returns:
        pd.DataFrame: Enhanced feature matrix
    """
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)
    
    X_eng = X.copy()
    
    # Age groups
    X_eng['age_group'] = pd.cut(X_eng['age'], 
                               bins=[0, 40, 50, 60, 100], 
                               labels=['young', 'middle_aged', 'senior', 'elderly'])
    X_eng['age_group'] = LabelEncoder().fit_transform(X_eng['age_group'])
    
    # BMI-like feature (if we had height/weight, we'd use actual BMI)
    # Using cholesterol to age ratio as a proxy for metabolic health
    X_eng['chol_age_ratio'] = X_eng['chol'] / X_eng['age']
    
    # Exercise capacity indicator
    # Higher thalach (max heart rate) relative to age indicates better fitness
    X_eng['heart_rate_reserve'] = X_eng['thalach'] - (220 - X_eng['age'])
    
    # Risk score based on multiple factors
    X_eng['risk_score'] = (
        X_eng['age'] * 0.1 +
        X_eng['chol'] * 0.01 +
        X_eng['trestbps'] * 0.1 +
        X_eng['oldpeak'] * 10
    )
    
    # Interaction features
    X_eng['age_sex_interaction'] = X_eng['age'] * X_eng['sex']
    X_eng['cp_exang_interaction'] = X_eng['cp'] * X_eng['exang']
    
    print(f"Original features: {X.shape[1]}")
    print(f"Engineered features: {X_eng.shape[1]}")
    print(f"New features added: {X_eng.shape[1] - X.shape[1]}")
    
    new_features = [col for col in X_eng.columns if col not in X.columns]
    print(f"New feature names: {new_features}")
    
    return X_eng

def create_preprocessing_pipeline():
    """
    Create preprocessing pipeline for features
    
    Returns:
        Pipeline: Scikit-learn preprocessing pipeline
    """
    print("\n" + "="*50)
    print("CREATING PREPROCESSING PIPELINE")
    print("="*50)
    
    # Use RobustScaler to handle outliers better than StandardScaler
    scaler = RobustScaler()
    
    preprocessing_pipeline = Pipeline([
        ('scaler', scaler)
    ])
    
    print("Preprocessing pipeline created with RobustScaler")
    
    return preprocessing_pipeline

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Test set proportion
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*50)
    print("SPLITTING DATA")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Test size: {test_size*100:.1f}%")
    
    print(f"\nTraining set target distribution:")
    print(y_train.value_counts().sort_index())
    
    print(f"\nTest set target distribution:")
    print(y_test.value_counts().sort_index())
    
    return X_train, X_test, y_train, y_test

def define_models():
    """
    Define multiple models for comparison
    
    Returns:
        dict: Dictionary of model instances
    """
    print("\n" + "="*50)
    print("DEFINING MODELS")
    print("="*50)
    
    models = {
        'logistic_regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            class_weight='balanced'
        ),
        'gradient_boosting': GradientBoostingClassifier(
            random_state=42,
            n_estimators=100
        ),
        'svm': SVC(
            random_state=42,
            probability=True,
            class_weight='balanced'
        )
    }
    
    print("Models defined:")
    for name, model in models.items():
        print(f"  - {name}: {type(model).__name__}")
    
    return models

def perform_cross_validation(models, X_train, y_train, cv_folds=5):
    """
    Perform cross-validation for all models
    
    Args:
        models (dict): Dictionary of models
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv_folds (int): Number of CV folds
        
    Returns:
        dict: Cross-validation results
    """
    print("\n" + "="*50)
    print("CROSS-VALIDATION EVALUATION")
    print("="*50)
    
    cv_results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('preprocessor', create_preprocessing_pipeline()),
            ('classifier', model)
        ])
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        
        cv_results[name] = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'pipeline': pipeline
        }
        
        print(f"  ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Sort by mean score
    sorted_results = sorted(cv_results.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"\nCross-validation ranking:")
    for i, (name, results) in enumerate(sorted_results, 1):
        print(f"  {i}. {name}: {results['mean']:.4f} (+/- {results['std'] * 2:.4f})")
    
    return cv_results

def hyperparameter_tuning(best_models, X_train, y_train):
    """
    Perform hyperparameter tuning for top models
    
    Args:
        best_models (dict): Top performing models from CV
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        dict: Tuned models
    """
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    tuned_models = {}
    
    # Define parameter grids
    param_grids = {
        'logistic_regression': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        },
        'random_forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'gradient_boosting': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds for speed
    
    for name, model_info in best_models.items():
        if name in param_grids:
            print(f"\nTuning {name}...")
            
            pipeline = model_info['pipeline']
            param_grid = param_grids[name]
            
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring='roc_auc', 
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            tuned_models[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            print(f"  Best score: {grid_search.best_score_:.4f}")
            print(f"  Best params: {grid_search.best_params_}")
        else:
            # Keep original model if no tuning defined
            tuned_models[name] = {
                'model': model_info['pipeline'],
                'best_score': model_info['mean']
            }
    
    return tuned_models

def evaluate_models(tuned_models, X_train, X_test, y_train, y_test):
    """
    Evaluate tuned models on test set
    
    Args:
        tuned_models (dict): Tuned models
        X_train, X_test (pd.DataFrame): Train/test features
        y_train, y_test (pd.Series): Train/test targets
        
    Returns:
        dict: Evaluation results
    """
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION")
    print("="*50)
    
    evaluation_results = {}
    
    for name, model_info in tuned_models.items():
        print(f"\nEvaluating {name}...")
        
        model = model_info['model']
        
        # Fit on training data
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        evaluation_results[name] = {
            'model': model,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return evaluation_results

def create_model_comparison_plots(evaluation_results, y_test):
    """
    Create comparison plots for model evaluation
    Note: All plots are commented for conversion to Jupyter notebook
    
    Args:
        evaluation_results (dict): Model evaluation results
        y_test (pd.Series): Test target values
    """
    print("\n" + "="*50)
    print("CREATING MODEL COMPARISON PLOTS")
    print("="*50)
    
    print("Creating model comparison visualizations (commented for Jupyter conversion)...")
    
    # 1. ROC Curves Comparison
    print("1. ROC curves comparison")
    # plt.figure(figsize=(10, 8))
    # 
    # for name, results in evaluation_results.items():
    #     y_pred_proba = results['y_pred_proba']
    #     fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    #     auc_score = results['metrics']['roc_auc']
    #     plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
    # 
    # plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves Comparison')
    # plt.legend(loc="lower right")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(FIGURES_DIR / 'roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 2. Precision-Recall Curves
    print("2. Precision-recall curves")
    # plt.figure(figsize=(10, 8))
    # 
    # for name, results in evaluation_results.items():
    #     y_pred_proba = results['y_pred_proba']
    #     precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    #     plt.plot(recall, precision, label=f'{name}')
    # 
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curves Comparison')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(FIGURES_DIR / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 3. Metrics Comparison Bar Plot
    print("3. Metrics comparison bar plot")
    # metrics_df = pd.DataFrame({
    #     name: results['metrics'] 
    #     for name, results in evaluation_results.items()
    # }).T
    # 
    # fig, ax = plt.subplots(figsize=(12, 8))
    # metrics_df.plot(kind='bar', ax=ax)
    # ax.set_title('Model Performance Metrics Comparison')
    # ax.set_ylabel('Score')
    # ax.set_xlabel('Models')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # ax.grid(True, alpha=0.3)
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(FIGURES_DIR / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 4. Confusion Matrices
    print("4. Confusion matrices")
    # n_models = len(evaluation_results)
    # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # axes = axes.flatten()
    # 
    # for i, (name, results) in enumerate(evaluation_results.items()):
    #     if i < len(axes):
    #         cm = results['confusion_matrix']
    #         sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
    #         axes[i].set_title(f'{name} Confusion Matrix')
    #         axes[i].set_xlabel('Predicted')
    #         axes[i].set_ylabel('Actual')
    # 
    # # Hide empty subplots
    # for i in range(len(evaluation_results), len(axes)):
    #     axes[i].set_visible(False)
    # 
    # plt.tight_layout()
    # plt.savefig(FIGURES_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("All model comparison plots created and saved to figures/ directory")

def save_models_and_results(evaluation_results, target_info, feature_names):
    """
    Save trained models and evaluation results
    
    Args:
        evaluation_results (dict): Model evaluation results
        target_info (dict): Target variable information
        feature_names (list): List of feature names
    """
    print("\n" + "="*50)
    print("SAVING MODELS AND RESULTS")
    print("="*50)
    
    # Find best model
    best_model_name = max(evaluation_results.keys(), 
                         key=lambda x: evaluation_results[x]['metrics']['roc_auc'])
    best_model = evaluation_results[best_model_name]['model']
    
    print(f"Best model: {best_model_name}")
    print(f"Best ROC-AUC: {evaluation_results[best_model_name]['metrics']['roc_auc']:.4f}")
    
    # Save best model
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved to: {best_model_path}")
    
    # Save all models
    for name, results in evaluation_results.items():
        model_path = MODELS_DIR / f"{name}_model.joblib"
        joblib.dump(results['model'], model_path)
        print(f"{name} model saved to: {model_path}")
    
    # Save evaluation results
    results_summary = {}
    for name, results in evaluation_results.items():
        results_summary[name] = {
            'metrics': results['metrics'],
            'classification_report': results['classification_report']
        }
    
    # Add metadata
    results_summary['metadata'] = {
        'best_model': best_model_name,
        'target_info': target_info,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = MODELS_DIR / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"Evaluation results saved to: {results_path}")
    
    # Save feature names for later use (only if changed)
    feature_names_path = MODELS_DIR / "feature_names.json"
    
    # Check if file exists and content is the same
    should_save = True
    if feature_names_path.exists():
        try:
            with open(feature_names_path, 'r') as f:
                existing_names = json.load(f)
            # Only save if content is different (preserve original order)
            should_save = existing_names != feature_names
        except (json.JSONDecodeError, FileNotFoundError):
            should_save = True
    
    if should_save:
        with open(feature_names_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"Feature names updated and saved to: {feature_names_path}")
    else:
        print(f"Feature names unchanged, skipping save: {feature_names_path}")

def main():
    """
    Main function to execute Task 2: Feature Engineering & Model Development
    """
    print("Starting Task 2: Feature Engineering & Model Development")
    print("="*70)
    
    try:
        # Step 1: Load processed data
        X, y = load_processed_data()
        
        # Step 2: Handle missing values
        X_clean = handle_missing_values(X)
        
        # Step 3: Prepare target variable
        y_binary, y_multiclass, target_info = prepare_target_variable(y)
        
        # Step 4: Feature engineering
        X_engineered = engineer_features(X_clean)
        
        # Step 5: Split data
        X_train, X_test, y_train, y_test = split_data(X_engineered, y_binary)
        
        # Step 6: Define models
        models = define_models()
        
        # Step 7: Cross-validation
        cv_results = perform_cross_validation(models, X_train, y_train)
        
        # Step 8: Select top models for tuning (top 3)
        top_models = dict(list(sorted(cv_results.items(), 
                                    key=lambda x: x[1]['mean'], 
                                    reverse=True))[:3])
        
        # Step 9: Hyperparameter tuning
        tuned_models = hyperparameter_tuning(top_models, X_train, y_train)
        
        # Step 10: Final evaluation
        evaluation_results = evaluate_models(tuned_models, X_train, X_test, y_train, y_test)
        
        # Step 11: Create comparison plots
        create_model_comparison_plots(evaluation_results, y_test)
        
        # Step 12: Save models and results
        save_models_and_results(evaluation_results, target_info, X_engineered.columns.tolist())
        
        print("\n" + "="*70)
        print("Task 2 completed successfully!")
        print("="*70)
        
        return evaluation_results, X_engineered, y_binary
        
    except Exception as e:
        print(f"Error in Task 2: {e}")
        raise

if __name__ == "__main__":
    evaluation_results, X_engineered, y_binary = main()