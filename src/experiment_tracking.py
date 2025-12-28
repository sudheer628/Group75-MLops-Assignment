"""
Task 3: MLflow Experiment Tracking
Heart Disease Prediction Model Experiment Tracking

This script handles:
1. MLflow experiment setup and configuration
2. Experiment logging with parameters, metrics, and artifacts
3. Model registration and versioning
4. Experiment comparison and analysis
5. Integration with Tasks 1 and 2
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import from previous tasks
try:
    from src.data_acquisition_eda import load_heart_disease_data
    from src.feature_engineering import (
        create_preprocessing_pipeline,
        engineer_features,
        handle_missing_values,
        prepare_target_variable,
        split_data,
    )
except ImportError:
    print("Warning: Could not import from previous tasks. Make sure to run Tasks 1 and 2 first.")

# Set up local directories (for models only)
MODELS_DIR = Path("models")
EXPERIMENTS_DIR = Path("experiments")

# Create directories if they don't exist (no mlruns - using Railway)
for directory in [MODELS_DIR, EXPERIMENTS_DIR]:
    directory.mkdir(exist_ok=True)


def setup_mlflow():
    """
    Setup MLflow tracking with Railway remote server

    Returns:
        str: MLflow tracking URI
    """
    print("\n" + "=" * 50)
    print("SETTING UP MLFLOW TRACKING")
    print("=" * 50)

    # Set Railway MLflow server as tracking URI
    railway_mlflow_uri = "https://mlflow-tracking-production-53fb.up.railway.app"
    mlflow.set_tracking_uri(railway_mlflow_uri)

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow server: Railway (Remote)")
    print(f"Artifacts will be stored on Railway server")

    return mlflow.get_tracking_uri()


def create_experiment(experiment_name, description=None):
    """
    Create or get MLflow experiment

    Args:
        experiment_name (str): Name of the experiment
        description (str): Description of the experiment

    Returns:
        str: Experiment ID
    """
    print(f"\nCreating/Getting experiment: {experiment_name}")

    try:
        # Try to get existing experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment ID: {experiment_id}")
        else:
            # Create new experiment
            experiment_id = mlflow.create_experiment(
                name=experiment_name, tags={"description": description or "Heart Disease Prediction Experiment"}
            )
            print(f"Created new experiment ID: {experiment_id}")

        # Set the experiment
        mlflow.set_experiment(experiment_name)

        return experiment_id

    except Exception as e:
        print(f"Error creating experiment: {e}")
        raise


def log_data_info(X, y, experiment_id):
    """
    Log dataset information as experiment metadata

    Args:
        X (pd.DataFrame): Features
        y (pd.DataFrame or pd.Series): Target
        experiment_id (str): MLflow experiment ID
    """
    print("\nLogging dataset information...")

    with mlflow.start_run(run_name="data_info") as run:
        # Handle both DataFrame and Series for target
        if isinstance(y, pd.DataFrame):
            target_series = y.iloc[:, 0]  # Get first column as Series
        else:
            target_series = y

        # Log dataset parameters
        mlflow.log_param("dataset_name", "Heart Disease UCI")
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", len(X.columns))
        mlflow.log_param("target_classes", len(target_series.unique()))

        # Log feature names
        mlflow.log_param("feature_names", list(X.columns))

        # Log data quality metrics
        missing_values = X.isnull().sum().sum()
        mlflow.log_metric("missing_values_count", missing_values)
        mlflow.log_metric("missing_values_percentage", (missing_values / (len(X) * len(X.columns))) * 100)

        # Log class distribution
        class_distribution = target_series.value_counts().to_dict()
        for class_label, count in class_distribution.items():
            mlflow.log_metric(f"class_{class_label}_count", count)
            mlflow.log_metric(f"class_{class_label}_percentage", (count / len(target_series)) * 100)

        # Log tags
        mlflow.set_tag("stage", "data_exploration")
        mlflow.set_tag("task", "Task_1_Data_Acquisition")

        print(f"Data info logged to run: {run.info.run_id}")


def log_feature_engineering_info(X_original, X_engineered, experiment_id):
    """
    Log feature engineering information

    Args:
        X_original (pd.DataFrame): Original features
        X_engineered (pd.DataFrame): Engineered features
        experiment_id (str): MLflow experiment ID
    """
    print("\nLogging feature engineering information...")

    with mlflow.start_run(run_name="feature_engineering") as run:
        # Log feature engineering parameters
        mlflow.log_param("original_features", len(X_original.columns))
        mlflow.log_param("engineered_features", len(X_engineered.columns))
        mlflow.log_param("new_features_added", len(X_engineered.columns) - len(X_original.columns))

        # Log new feature names
        new_features = [col for col in X_engineered.columns if col not in X_original.columns]
        mlflow.log_param("new_feature_names", new_features)

        # Log feature engineering techniques
        mlflow.log_param(
            "techniques_used", ["age_grouping", "ratio_features", "interaction_features", "derived_features", "robust_scaling"]
        )

        # Log tags
        mlflow.set_tag("stage", "feature_engineering")
        mlflow.set_tag("task", "Task_2_Feature_Engineering")

        print(f"Feature engineering info logged to run: {run.info.run_id}")


def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test, cv_scores=None, hyperparameters=None):
    """
    Train model and log everything to MLflow

    Args:
        model_name (str): Name of the model
        model: Scikit-learn model or pipeline
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        cv_scores (dict): Cross-validation scores
        hyperparameters (dict): Model hyperparameters

    Returns:
        dict: Model evaluation results
    """
    print(f"\nTraining and logging model: {model_name}")

    with mlflow.start_run(run_name=f"{model_name}_training") as run:

        # Log model parameters
        if hyperparameters:
            for param, value in hyperparameters.items():
                mlflow.log_param(param, value)

        # Log model type
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("algorithm", type(model).__name__)

        # Train the model
        print(f"  Training {model_name}...")
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }

        if y_pred_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log cross-validation scores if available
        if cv_scores:
            mlflow.log_metric("cv_mean_score", cv_scores["mean"])
            mlflow.log_metric("cv_std_score", cv_scores["std"])
            # Log individual fold scores with steps for visualization
            for i, score in enumerate(cv_scores["scores"]):
                mlflow.log_metric("cv_fold_score", score, step=i + 1)
                mlflow.log_metric(f"cv_fold_{i + 1}_score", score)

        # Log dataset info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("train_test_split", "80/20")

        # Create and log confusion matrix plot
        # cm_fig = create_confusion_matrix_plot(y_test, y_pred, model_name)
        # mlflow.log_figure(cm_fig, f"{model_name}_confusion_matrix.png")
        # plt.close(cm_fig)

        # Log model using Railway 2.10.0 compatible approach (avoid new MLflow 3.x APIs)
        try:
            # Alternative approach: Save model locally and log as generic artifact
            import os
            import tempfile

            # Create temporary directory for model
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "model")

                # Save model using older MLflow API (no new endpoints)
                mlflow.sklearn.save_model(model, model_path)

                # Log the saved model directory as artifact
                mlflow.log_artifacts(model_path, "model")

                print(f"  Model artifacts logged successfully (Railway 2.10.0 compatible)")

                # Try model registration using older API if needed
                try:
                    # Get the artifact URI for registration
                    run = mlflow.active_run()
                    model_uri = f"runs:/{run.info.run_id}/model"

                    registered_model_name = f"heart_disease_{model_name}"
                    mlflow.register_model(model_uri=model_uri, name=registered_model_name)
                    print(f"  Model registered successfully: {registered_model_name}")
                except Exception as reg_e:
                    print(f"  Model registration failed (continuing without registry): {reg_e}")
                    # Continue - model artifacts are still logged successfully
                    pass

        except Exception as e:
            print(f"  Model logging failed: {e}")
            # Continue without model logging
            pass

        # Log additional artifacts
        # Save classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        class_report_path = f"classification_report_{model_name}.json"
        with open(class_report_path, "w") as f:
            json.dump(class_report, f, indent=2)
        mlflow.log_artifact(class_report_path)
        os.remove(class_report_path)  # Clean up temporary file

        # Log tags
        mlflow.set_tag("stage", "model_training")
        mlflow.set_tag("task", "Task_2_Model_Development")
        mlflow.set_tag("model_family", model_name.split("_")[0])

        # Add performance tier tag
        if metrics.get("roc_auc", 0) > 0.9:
            mlflow.set_tag("performance_tier", "excellent")
        elif metrics.get("roc_auc", 0) > 0.8:
            mlflow.set_tag("performance_tier", "good")
        else:
            mlflow.set_tag("performance_tier", "needs_improvement")

        print(f"  Model logged to run: {run.info.run_id}")
        print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")

        return {
            "run_id": run.info.run_id,
            "metrics": metrics,
            "model": model,
            "predictions": y_pred,
            "probabilities": y_pred_proba,
        }


def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """
    Create confusion matrix plot (commented for Jupyter notebook)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model

    Returns:
        matplotlib.figure.Figure: Confusion matrix plot
    """
    print(f"Creating confusion matrix plot for {model_name} (commented for Jupyter notebook)")

    # fig, ax = plt.subplots(figsize=(8, 6))
    # cm = confusion_matrix(y_true, y_pred)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    # ax.set_title(f'Confusion Matrix - {model_name}')
    # ax.set_xlabel('Predicted')
    # ax.set_ylabel('Actual')
    # plt.tight_layout()
    # return fig

    # Return dummy figure for now
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.text(0.5, 0.5, f"Confusion Matrix\n{model_name}", ha="center", va="center")
    return fig


def run_experiment_comparison(experiment_name="heart_disease_comparison"):
    """
    Run complete experiment with multiple models and log everything

    Args:
        experiment_name (str): Name of the MLflow experiment

    Returns:
        dict: Experiment results
    """
    print("Starting MLflow Experiment: Model Comparison")
    print("=" * 60)

    # Setup MLflow
    setup_mlflow()
    experiment_id = create_experiment(experiment_name, "Comprehensive heart disease prediction model comparison")

    try:
        # Step 1: Load and prepare data
        print("\nStep 1: Loading and preparing data...")
        X, y = load_heart_disease_data()[:2]  # Get X, y (ignore metadata)

        # Log original data info
        log_data_info(X, y, experiment_id)

        # Step 2: Feature engineering
        print("\nStep 2: Feature engineering...")
        X_clean = handle_missing_values(X)
        y_binary, _, _ = prepare_target_variable(y)
        X_engineered = engineer_features(X_clean)

        # Log feature engineering info
        log_feature_engineering_info(X_clean, X_engineered, experiment_id)

        # Step 3: Split data
        X_train, X_test, y_train, y_test = split_data(X_engineered, y_binary)

        # Step 4: Define models with hyperparameters
        models_config = {
            "logistic_regression": {
                "model": Pipeline(
                    [("scaler", RobustScaler()), ("classifier", LogisticRegression(random_state=42, max_iter=1000))]
                ),
                "hyperparameters": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000},
            },
            "random_forest": {
                "model": Pipeline(
                    [("scaler", RobustScaler()), ("classifier", RandomForestClassifier(random_state=42, n_estimators=100))]
                ),
                "hyperparameters": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
            },
            "gradient_boosting": {
                "model": Pipeline([("scaler", RobustScaler()), ("classifier", GradientBoostingClassifier(random_state=42))]),
                "hyperparameters": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
            },
            "svm": {
                "model": Pipeline([("scaler", RobustScaler()), ("classifier", SVC(random_state=42, probability=True))]),
                "hyperparameters": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
            },
        }

        # Step 5: Train and log all models
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, config in models_config.items():
            print(f"\nProcessing {model_name}...")

            # Perform cross-validation
            cv_scores = cross_val_score(config["model"], X_train, y_train, cv=cv, scoring="roc_auc")
            cv_results = {"scores": cv_scores, "mean": cv_scores.mean(), "std": cv_scores.std()}

            # Train and log model
            result = train_and_log_model(
                model_name=model_name,
                model=config["model"],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                cv_scores=cv_results,
                hyperparameters=config["hyperparameters"],
            )

            results[model_name] = result

        # Step 6: Log experiment summary
        log_experiment_summary(results, experiment_id)

        print(f"\n" + "=" * 60)
        print("MLflow Experiment completed successfully!")
        print(f"Experiment ID: {experiment_id}")
        print(f"MLflow UI: https://mlflow-tracking-production-53fb.up.railway.app")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print("=" * 60)

        return {"experiment_id": experiment_id, "results": results, "tracking_uri": mlflow.get_tracking_uri()}

    except Exception as e:
        print(f"Error in experiment: {e}")
        raise


def log_experiment_summary(results, experiment_id):
    """
    Log experiment summary with model comparison

    Args:
        results (dict): Results from all models
        experiment_id (str): MLflow experiment ID
    """
    print("\nLogging experiment summary...")

    with mlflow.start_run(run_name="experiment_summary") as run:

        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]["metrics"].get("roc_auc", 0))
        best_roc_auc = results[best_model_name]["metrics"]["roc_auc"]

        # Log summary metrics
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_roc_auc", best_roc_auc)
        mlflow.log_param("models_compared", len(results))
        mlflow.log_param("experiment_timestamp", datetime.now().isoformat())

        # Log all model performances
        for model_name, result in results.items():
            metrics = result["metrics"]
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)

        # Create model comparison summary
        comparison_summary = {}
        for model_name, result in results.items():
            comparison_summary[model_name] = {
                "roc_auc": result["metrics"].get("roc_auc", 0),
                "accuracy": result["metrics"]["accuracy"],
                "f1_score": result["metrics"]["f1_score"],
                "run_id": result["run_id"],
            }

        # Save comparison summary as artifact
        summary_path = "model_comparison_summary.json"
        with open(summary_path, "w") as f:
            json.dump(comparison_summary, f, indent=2)
        mlflow.log_artifact(summary_path)
        os.remove(summary_path)  # Clean up

        # Log tags
        mlflow.set_tag("stage", "experiment_summary")
        mlflow.set_tag("task", "Task_3_Experiment_Tracking")
        mlflow.set_tag("experiment_type", "model_comparison")

        print(f"Experiment summary logged to run: {run.info.run_id}")
        print(f"Best model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")

        # Set model aliases for best model
        try:
            from mlflow import MlflowClient

            client = MlflowClient()

            # Set champion alias for best model
            registered_model_name = f"heart_disease_{best_model_name}"
            latest_version = client.get_latest_versions(registered_model_name)[0].version

            client.set_registered_model_alias(name=registered_model_name, alias="champion", version=latest_version)

            print(f"Set 'champion' alias for {registered_model_name} version {latest_version}")

        except Exception as e:
            print(f"Could not set model alias: {e}")


def view_experiment_results(experiment_name="heart_disease_comparison"):
    """
    View and analyze experiment results from MLflow

    Args:
        experiment_name (str): Name of the experiment to analyze
    """
    print("\n" + "=" * 50)
    print("VIEWING EXPERIMENT RESULTS")
    print("=" * 50)

    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Experiment '{experiment_name}' not found!")
            return

        # Get all runs from the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            print("No runs found in the experiment!")
            return

        print(f"Experiment: {experiment_name}")
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Total runs: {len(runs)}")

        # Filter model training runs
        model_runs = runs[runs["tags.stage"] == "model_training"]

        if not model_runs.empty:
            print(f"\nModel Training Runs ({len(model_runs)}):")
            print("-" * 80)

            # Sort by ROC-AUC if available
            if "metrics.roc_auc" in model_runs.columns:
                model_runs = model_runs.sort_values("metrics.roc_auc", ascending=False)

            for _, run in model_runs.iterrows():
                model_type = run.get("params.model_type", "Unknown")
                roc_auc = run.get("metrics.roc_auc", "N/A")
                accuracy = run.get("metrics.accuracy", "N/A")
                run_id = run["run_id"][:8]

                # Handle formatting for numeric vs string values
                if isinstance(roc_auc, (int, float)):
                    roc_auc_str = f"{roc_auc:6.4f}"
                else:
                    roc_auc_str = f"{str(roc_auc):>6}"

                if isinstance(accuracy, (int, float)):
                    accuracy_str = f"{accuracy:6.4f}"
                else:
                    accuracy_str = f"{str(accuracy):>6}"

                print(f"  {model_type:20} | ROC-AUC: {roc_auc_str} | Accuracy: {accuracy_str} | Run: {run_id}")

        # Show experiment summary if available
        summary_runs = runs[runs["tags.stage"] == "experiment_summary"]
        if not summary_runs.empty:
            summary_run = summary_runs.iloc[0]
            best_model = summary_run.get("params.best_model", "Unknown")
            best_roc_auc = summary_run.get("metrics.best_roc_auc", "N/A")

            print(f"\nExperiment Summary:")
            print(f"  Best Model: {best_model}")
            if isinstance(best_roc_auc, (int, float)):
                print(f"  Best ROC-AUC: {best_roc_auc:.4f}")
            else:
                print(f"  Best ROC-AUC: {best_roc_auc}")

        print(f"\nTo view in MLflow UI:")
        print(f"  Open: https://mlflow-tracking-production-53fb.up.railway.app")
        print(f"  (Railway MLflow Server - accessible to all teammates)")

    except Exception as e:
        print(f"Error viewing experiment results: {e}")


def main():
    """
    Main function to execute Task 3: MLflow Experiment Tracking
    """
    print("Starting Task 3: MLflow Experiment Tracking")
    print("=" * 70)

    try:
        # Run the complete experiment
        experiment_results = run_experiment_comparison()

        # View results
        view_experiment_results()

        print("\n" + "=" * 70)
        print("Task 3 completed successfully!")
        print("=" * 70)

        return experiment_results

    except Exception as e:
        print(f"Error in Task 3: {e}")
        raise


if __name__ == "__main__":
    experiment_results = main()
