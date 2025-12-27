"""
CI/CD utilities for GitHub Actions compatibility
Provides environment setup, reporting, and CI-specific functionality
"""

import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path


def setup_ci_environment():
    """
    Setup environment for CI/CD execution
    Creates required directories and sets environment variables
    """
    print("Setting up CI/CD environment...")

    # Create required directories with proper structure
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "models/experiments",
        "models/production",
        "models/validation",
        "models/archived",
        "logs",
        "figures",
        "packages",
        "environments",
        "configs",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Create .gitkeep files to preserve directory structure
    gitkeep_dirs = ["data/raw", "data/processed", "logs", "figures", "packages"]
    for dir_path in gitkeep_dirs:
        gitkeep_file = Path(dir_path) / ".gitkeep"
        gitkeep_file.touch(exist_ok=True)

    # Set MLflow tracking URI from environment or default
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow-tracking-production-53fb.up.railway.app")
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    print(f"MLflow URI set to: {mlflow_uri}")

    # Set Python path for imports
    current_dir = str(Path.cwd())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"Added to Python path: {current_dir}")

    # CI-specific environment variables
    if os.getenv("CI"):
        os.environ["PYTHONUNBUFFERED"] = "1"  # Ensure output is not buffered
        os.environ["MPLBACKEND"] = "Agg"  # Use non-interactive matplotlib backend
        print("CI-specific environment variables set")

    print("CI/CD environment setup completed successfully")
    return True


def generate_ci_report():
    """
    Generate comprehensive CI/CD execution report
    Includes environment info, execution context, and results
    """
    print("Generating CI/CD execution report...")

    # Collect system information
    report = {
        "execution_info": {
            "timestamp": datetime.now().isoformat(),
            "working_directory": str(Path.cwd()),
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
        },
        "environment_variables": {
            "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "Not set"),
            "PYTHONPATH": os.getenv("PYTHONPATH", "Not set"),
            "CI": os.getenv("CI", "false"),
            "GITHUB_WORKFLOW": os.getenv("GITHUB_WORKFLOW", "Not in GitHub Actions"),
            "GITHUB_RUN_ID": os.getenv("GITHUB_RUN_ID", "Not in GitHub Actions"),
            "GITHUB_SHA": os.getenv("GITHUB_SHA", "Not in GitHub Actions"),
            "GITHUB_REF": os.getenv("GITHUB_REF", "Not in GitHub Actions"),
        },
        "directory_structure": {},
        "file_counts": {},
    }

    # Check directory structure and file counts
    check_dirs = ["data", "models", "logs", "figures", "packages", "src", "tests"]
    for dir_name in check_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            files = list(dir_path.rglob("*"))
            report["directory_structure"][dir_name] = "exists"
            report["file_counts"][dir_name] = len([f for f in files if f.is_file()])
        else:
            report["directory_structure"][dir_name] = "missing"
            report["file_counts"][dir_name] = 0

    # Check for key files
    key_files = [
        "requirements.txt",
        "README.md",
        "run_tests.py",
        "src/data_acquisition_eda.py",
        "src/feature_engineering.py",
        "src/experiment_tracking.py",
        "src/model_packaging.py",
    ]

    report["key_files"] = {}
    for file_path in key_files:
        report["key_files"][file_path] = Path(file_path).exists()

    # Save report to logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    report_file = logs_dir / "ci_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"CI/CD report saved to: {report_file}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("CI/CD EXECUTION REPORT SUMMARY")
    print("=" * 60)
    print(f"Timestamp: {report['execution_info']['timestamp']}")
    print(f"Python Version: {report['execution_info']['python_version'].split()[0]}")
    print(f"Platform: {report['execution_info']['platform']}")
    print(f"CI Environment: {report['environment_variables']['CI']}")
    print(f"GitHub Workflow: {report['environment_variables']['GITHUB_WORKFLOW']}")
    print(f"MLflow URI: {report['environment_variables']['MLFLOW_TRACKING_URI']}")

    print("\nDirectory Status:")
    for dir_name, status in report["directory_structure"].items():
        file_count = report["file_counts"][dir_name]
        print(f"  {dir_name}: {status} ({file_count} files)")

    print("\nKey Files Status:")
    for file_path, exists in report["key_files"].items():
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")

    print("=" * 60)

    return report


def validate_ci_environment():
    """
    Validate CI/CD environment setup
    Returns True if environment is properly configured
    """
    print("Validating CI/CD environment...")

    validation_results = {
        "python_version": False,
        "required_directories": False,
        "key_files": False,
        "mlflow_uri": False,
        "imports": False,
    }

    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 10):
        validation_results["python_version"] = True
        print(f"Python version: {python_version.major}.{python_version.minor}")
    else:
        print(f"Python version too old: {python_version.major}.{python_version.minor}")

    # Check required directories
    required_dirs = ["src", "tests", "data", "models"]
    missing_dirs = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)

    if not missing_dirs:
        validation_results["required_directories"] = True
        print("Required directories exist")
    else:
        print(f"Missing directories: {missing_dirs}")

    # Check key files
    key_files = ["requirements.txt", "run_tests.py"]
    missing_files = []
    for file_path in key_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if not missing_files:
        validation_results["key_files"] = True
        print("Key files exist")
    else:
        print(f"Missing files: {missing_files}")

    # Check MLflow URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri and "railway.app" in mlflow_uri:
        validation_results["mlflow_uri"] = True
        print("MLflow URI configured")
    else:
        print("MLflow URI not properly configured")

    # Check critical imports
    try:
        import mlflow
        import numpy
        import pandas
        import sklearn

        validation_results["imports"] = True
        print("Critical packages can be imported")
    except ImportError as e:
        print(f"Import error: {e}")

    # Summary
    all_passed = all(validation_results.values())
    if all_passed:
        print("\nCI/CD environment validation PASSED")
    else:
        print("\nCI/CD environment validation FAILED")
        failed_checks = [k for k, v in validation_results.items() if not v]
        print(f"Failed checks: {failed_checks}")

    return all_passed


def log_ci_step(step_name, status="started", details=None):
    """
    Log CI/CD pipeline step with timestamp
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if status == "started":
        print(f"\n[{timestamp}] Starting: {step_name}")
    elif status == "completed":
        print(f"[{timestamp}] Completed: {step_name}")
    elif status == "failed":
        print(f"[{timestamp}] Failed: {step_name}")
    elif status == "skipped":
        print(f"[{timestamp}] Skipped: {step_name}")

    if details:
        print(f"    Details: {details}")


def get_ci_context():
    """
    Get CI/CD execution context information
    """
    return {
        "is_ci": os.getenv("CI", "false").lower() == "true",
        "is_github_actions": os.getenv("GITHUB_ACTIONS", "false").lower() == "true",
        "workflow_name": os.getenv("GITHUB_WORKFLOW"),
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "sha": os.getenv("GITHUB_SHA"),
        "ref": os.getenv("GITHUB_REF"),
        "event_name": os.getenv("GITHUB_EVENT_NAME"),
        "actor": os.getenv("GITHUB_ACTOR"),
    }


if __name__ == "__main__":
    """
    Allow running CI utilities directly for testing
    """
    print("MLOps CI/CD Utilities")
    print("=" * 40)

    # Setup environment
    setup_ci_environment()

    # Validate environment
    validate_ci_environment()

    # Generate report
    generate_ci_report()

    # Show CI context
    context = get_ci_context()
    print(f"\nCI Context: {context}")
