#!/usr/bin/env python3
"""
CI/CD Pipeline Validation Script
Validates that all CI/CD components are properly configured

Usage:
    python tests/validate_cicd.py
    OR from root: python -m tests.validate_cicd
"""

import json
import os
import sys
import yaml
from pathlib import Path


def get_root_dir():
    """Get the project root directory"""
    # If running from tests/, go up one level
    script_dir = Path(__file__).parent
    if script_dir.name == "tests":
        return script_dir.parent
    # If running from root, use current directory
    return Path.cwd()


def validate_workflow_files():
    """Validate GitHub Actions workflow files"""
    print("Validating GitHub Actions workflows...")

    root = get_root_dir()
    workflows_dir = root / ".github" / "workflows"

    if not workflows_dir.exists():
        print(".github/workflows directory not found")
        return False

    required_workflows = ["ci.yml", "model-training.yml", "pr-validation.yml"]
    missing_workflows = []

    for workflow in required_workflows:
        workflow_path = workflows_dir / workflow
        if not workflow_path.exists():
            missing_workflows.append(workflow)
        else:
            # Validate YAML syntax
            try:
                with open(workflow_path, "r") as f:
                    yaml.safe_load(f)
                print(f"{workflow} - Valid YAML")
            except yaml.YAMLError as e:
                print(f"{workflow} - Invalid YAML: {e}")
                return False

    if missing_workflows:
        print(f"Missing workflows: {missing_workflows}")
        return False

    print("All GitHub Actions workflows are valid")
    return True


def validate_configuration_files():
    """Validate configuration files"""
    print("\nValidating configuration files...")

    root = get_root_dir()
    config_files = {
        ".flake8": "Flake8 configuration",
        "pyproject.toml": "Black/isort configuration",
        "requirements.txt": "Python dependencies",
    }

    all_valid = True

    for file_path, description in config_files.items():
        if (root / file_path).exists():
            print(f"{file_path} - {description}")
        else:
            print(f"{file_path} - Missing {description}")
            all_valid = False

    return all_valid


def validate_ci_utilities():
    """Validate CI/CD utilities"""
    print("\nValidating CI/CD utilities...")

    root = get_root_dir()
    ci_utils_path = root / "src" / "ci_utils.py"

    if not ci_utils_path.exists():
        print("src/ci_utils.py not found")
        return False

    # Test import
    try:
        sys.path.insert(0, str(root))
        from src.ci_utils import (
            generate_ci_report,
            setup_ci_environment,
            validate_ci_environment,
        )

        print("CI utilities can be imported")

        # Test basic functionality
        setup_ci_environment()
        print("CI environment setup works")

        validate_ci_environment()
        print("CI environment validation works")

        return True

    except Exception as e:
        print(f"CI utilities error: {e}")
        return False


def validate_test_runner():
    """Validate test runner CI mode"""
    print("\nValidating test runner CI mode...")

    root = get_root_dir()
    run_tests_path = root / "run_tests.py"

    if not run_tests_path.exists():
        print("run_tests.py not found")
        return False

    # Check if CI mode is implemented
    with open(run_tests_path, "r") as f:
        content = f.read()

    if "--ci" in content and "run_ci_tests" in content:
        print("CI mode implemented in test runner")
        return True
    else:
        print("CI mode not properly implemented")
        return False


def validate_dependencies():
    """Validate CI/CD dependencies"""
    print("\nValidating CI/CD dependencies...")

    root = get_root_dir()
    required_packages = ["black", "isort", "flake8", "pytest"]

    with open(root / "requirements.txt", "r") as f:
        requirements = f.read()

    missing_packages = []
    for package in required_packages:
        if package not in requirements:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing CI/CD packages: {missing_packages}")
        return False

    print("All CI/CD dependencies are listed")
    return True


def validate_directory_structure():
    """Validate required directory structure"""
    print("\nValidating directory structure...")

    root = get_root_dir()
    required_dirs = [".github/workflows", "src", "tests", "data", "models"]

    missing_dirs = []
    for dir_path in required_dirs:
        if not (root / dir_path).exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"Missing directories: {missing_dirs}")
        return False

    print("Directory structure is correct")
    return True


def generate_validation_report():
    """Generate validation report"""
    print("\n" + "=" * 60)
    print("CI/CD VALIDATION REPORT")
    print("=" * 60)

    validations = [
        ("GitHub Actions Workflows", validate_workflow_files),
        ("Configuration Files", validate_configuration_files),
        ("CI/CD Utilities", validate_ci_utilities),
        ("Test Runner CI Mode", validate_test_runner),
        ("Dependencies", validate_dependencies),
        ("Directory Structure", validate_directory_structure),
    ]

    results = {}
    all_passed = True

    for name, validator in validations:
        try:
            result = validator()
            results[name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"{name} validation failed with error: {e}")
            results[name] = False
            all_passed = False

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name:30} {status}")

    print("=" * 60)

    if all_passed:
        print("ALL VALIDATIONS PASSED!")
        print("Your CI/CD pipeline is ready for GitHub Actions.")
        print("\nNext steps:")
        print("1. Commit and push your changes")
        print("2. Configure GitHub Secrets (MLFLOW_TRACKING_URI)")
        print("3. Create a pull request to test the pipeline")
        print("4. Monitor the Actions tab for workflow execution")
    else:
        print("SOME VALIDATIONS FAILED!")
        print("Please fix the issues above before deploying to GitHub Actions.")

    # Save report
    report = {
        "timestamp": str(Path.cwd()),
        "validation_results": results,
        "overall_status": "PASS" if all_passed else "FAIL",
    }

    print(f"\nValidation report: {report}")

    return all_passed


if __name__ == "__main__":
    print("MLOps CI/CD Pipeline Validation")
    print("=" * 40)

    # Run validation
    success = generate_validation_report()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
