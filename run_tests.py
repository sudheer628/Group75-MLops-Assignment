"""
Comprehensive Test Runner & Validation for MLOps Heart Disease Project

This script combines environment validation and test execution:
1. Validates environment setup (conda, dependencies, MLflow connection)
2. Checks for required files and directories
3. Runs comprehensive test suite for all tasks

Usage:
    python run_tests.py                    # Full validation + all tests
    python run_tests.py --task1           # Run only Task 1 tests
    python run_tests.py --task2           # Run only Task 2 tests  
    python run_tests.py --task3           # Run only Task 3 tests
    python run_tests.py --task4           # Run only Task 4 tests
    python run_tests.py --validate-only   # Run only environment validation
    python run_tests.py --skip-validation # Skip validation, run tests only
    python run_tests.py --pytest          # Run with pytest
"""

import sys
import os
import argparse
from pathlib import Path

def print_step(step_num, description):
    """Print formatted step header"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print('='*60)

def validate_environment():
    """Validate environment setup"""
    print_step("ENV", "Environment Validation")
    
    validation_results = {}
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env != 'myenv':
        print("ERROR: conda environment 'myenv' not activated")
        print("Please run: conda activate myenv")
        validation_results['environment'] = False
    else:
        print(f"Conda environment: {conda_env}")
        print(f"Python version: {sys.version.split()[0]}")
        validation_results['environment'] = True
    
    # Check dependencies
    print("\nChecking Dependencies:")
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sklearn', 'mlflow', 'ucimlrepo', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package}")
        except ImportError:
            print(f"{package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Run: pip install -r requirements.txt")
        validation_results['dependencies'] = False
    else:
        validation_results['dependencies'] = True
    
    # Check MLflow connection
    print("\nTesting Railway MLflow Connection:")
    try:
        import mlflow
        mlflow.set_tracking_uri("https://mlflow-tracking-production-53fb.up.railway.app")
        
        client = mlflow.MlflowClient()
        experiments = client.search_experiments()
        
        print(f"Connected to Railway MLflow server")
        print(f"Found {len(experiments)} experiments")
        validation_results['mlflow'] = True
        
    except Exception as e:
        print(f"MLflow connection failed: {e}")
        validation_results['mlflow'] = False
    
    # Check required directories and files
    print("\nChecking Project Structure:")
    required_paths = [
        'src/',
        'tests/', 
        'data/',
        'models/',
        'requirements.txt',
        'src/data_acquisition_eda.py',
        'src/feature_engineering.py',
        'src/experiment_tracking.py',
        'src/model_packaging.py'
    ]
    
    missing_paths = []
    for path in required_paths:
        if Path(path).exists():
            print(f"{path}")
        else:
            print(f"{path}")
            missing_paths.append(path)
    
    if missing_paths:
        print(f"\nMissing paths: {missing_paths}")
        validation_results['structure'] = False
    else:
        validation_results['structure'] = True
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY:")
    all_passed = True
    for check, passed in validation_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{check.capitalize():15} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nEnvironment validation successful!")
        print("Ready to run tests...")
    else:
        print("\nEnvironment validation failed!")
        print("Please fix the issues above before running tests.")
    
    return all_passed
    """Run Task 1 tests"""
    print("Running Task 1 Tests...")
    print("=" * 60)
    
    try:
        from tests.test_task1_data_acquisition import TestTask1DataAcquisition, test_full_task1_pipeline
        
        # Create test instance and run tests
        test_instance = TestTask1DataAcquisition()
        
        # Run individual tests
        test_instance.test_load_heart_disease_data()
        test_instance.test_data_quality_check()
        test_instance.test_eda_analysis()
        test_instance.test_save_processed_data()
        test_instance.test_data_consistency()
        
        # Run full pipeline test
        test_full_task1_pipeline()
        
        print("\n" + "=" * 60)
        print("All Task 1 tests passed!")
        return True
        
    except Exception as e:
        print(f"\nTask 1 tests failed: {e}")
        return False

def run_task1_tests():
    """Run Task 1 tests"""
    print("Running Task 1 Tests...")
    print("=" * 60)
    
    try:
        from tests.test_task1_data_acquisition import TestTask1DataAcquisition, test_full_task1_pipeline
        
        # Create test instance and run tests
        test_instance = TestTask1DataAcquisition()
        
        # Run individual tests
        test_instance.test_load_heart_disease_data()
        test_instance.test_data_quality_check()
        test_instance.test_eda_analysis()
        test_instance.test_save_processed_data()
        test_instance.test_data_consistency()
        
        # Run full pipeline test
        test_full_task1_pipeline()
        
        print("\n" + "=" * 60)
        print("All Task 1 tests passed!")
        return True
        
    except Exception as e:
        print(f"\nTask 1 tests failed: {e}")
        return False

def run_task2_tests():
    """Run Task 2 tests"""
    print("Running Task 2 Tests...")
    print("=" * 60)
    
    try:
        from tests.test_task2_feature_engineering import TestTask2FeatureEngineering, test_full_task2_pipeline
        
        # Create test instance and run tests
        test_instance = TestTask2FeatureEngineering()
        
        # Run individual tests
        test_instance.test_load_processed_data()
        test_instance.test_handle_missing_values()
        test_instance.test_prepare_target_variable()
        test_instance.test_engineer_features()
        test_instance.test_preprocessing_pipeline()
        test_instance.test_data_splitting()
        test_instance.test_model_definition()
        test_instance.test_cross_validation()
        test_instance.test_model_evaluation()
        
        # Run full pipeline test
        test_full_task2_pipeline()
        
        print("\n" + "=" * 60)
        print("All Task 2 tests passed!")
        return True
        
    except Exception as e:
        print(f"\nTask 2 tests failed: {e}")
        return False

def run_task3_tests():
    print("Running Task 3 Tests...")
    print("=" * 60)
    
    try:
        from tests.test_task3_experiment_tracking import TestTask3ExperimentTracking, test_full_task3_pipeline
        
        # Create test instance and run tests
        test_instance = TestTask3ExperimentTracking()
        
        # Run individual tests
        test_instance.test_mlflow_setup()
        test_instance.test_experiment_creation()
        test_instance.test_data_logging()
        test_instance.test_feature_engineering_logging()
        test_instance.test_model_logging()
        test_instance.test_experiment_viewing()
        test_instance.test_railway_server_connection()
        
        # Run full pipeline test
        test_full_task3_pipeline()
        
        print("\n" + "=" * 60)
        print("All Task 3 tests passed!")
        return True
        
    except Exception as e:
        print(f"\nTask 3 tests failed: {e}")
        return False

def run_task4_tests():
    """Run Task 4 tests"""
    print("Running Task 4 Tests...")
    print("=" * 60)
    
    try:
        from tests.test_task4_model_packaging import TestTask4ModelPackaging, test_full_task4_pipeline
        
        # Create test instance and run tests
        test_instance = TestTask4ModelPackaging()
        
        # Setup test environment
        test_instance.setup_test_environment()
        
        # Run individual tests
        if test_instance.test_mlflow_connection():
            test_instance.test_champion_model_identification()
            test_instance.test_preprocessing_pipeline_creation()
        
        test_instance.test_custom_transformers()
        test_instance.test_environment_snapshot()
        test_instance.test_configuration_files()
        test_instance.test_package_validation()
        
        # Run full pipeline test
        test_full_task4_pipeline()
        
        print("\n" + "=" * 60)
        print("All Task 4 tests passed!")
        return True
        
    except Exception as e:
        print(f"\nTask 4 tests failed: {e}")
        return False

def run_task2_tests():
    print("Running Task 2 Tests...")
    print("=" * 60)
    
    try:
        from tests.test_task2_feature_engineering import TestTask2FeatureEngineering, test_full_task2_pipeline
        
        # Create test instance and run tests
        test_instance = TestTask2FeatureEngineering()
        
        # Run individual tests
        test_instance.test_load_processed_data()
        test_instance.test_handle_missing_values()
        test_instance.test_prepare_target_variable()
        test_instance.test_engineer_features()
        test_instance.test_preprocessing_pipeline()
        test_instance.test_data_splitting()
        test_instance.test_model_definition()
        test_instance.test_cross_validation()
        test_instance.test_model_evaluation()
        
        # Run full pipeline test
        test_full_task2_pipeline()
        
        print("\n" + "=" * 60)
        print("All Task 2 tests passed!")
        return True
        
    except Exception as e:
        print(f"\nTask 2 tests failed: {e}")
        return False

def run_pytest():
    """Run tests using pytest"""
    import subprocess
    
    print("Running tests with pytest...")
    print("=" * 60)
    
    try:
        # Run pytest on the tests directory
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/", "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("All pytest tests passed!")
            return True
        else:
            print("Some pytest tests failed!")
            return False
            
    except FileNotFoundError:
        print("pytest not found. Install with: pip install pytest")
        return False
    except Exception as e:
        print(f"pytest execution failed: {e}")
        return False

def run_ci_tests():
    """Run tests in CI/CD mode with artifact generation"""
    try:
        from src.ci_utils import setup_ci_environment, generate_ci_report, validate_ci_environment, log_ci_step
        
        log_ci_step("CI/CD Test Pipeline", "started")
        
        # Setup CI environment
        log_ci_step("Environment Setup", "started")
        setup_ci_environment()
        log_ci_step("Environment Setup", "completed")
        
        # Validate CI environment (skip conda check in CI)
        log_ci_step("Environment Validation", "started")
        if not validate_ci_environment():
            log_ci_step("Environment Validation", "failed", "CI environment validation failed")
            return False
        log_ci_step("Environment Validation", "completed")
        
        # Run all tests
        log_ci_step("Test Execution", "started")
        success = True
        
        log_ci_step("Task 1 Tests", "started")
        task1_success = run_task1_tests()
        if task1_success:
            log_ci_step("Task 1 Tests", "completed")
        else:
            log_ci_step("Task 1 Tests", "failed")
            success = False
        
        log_ci_step("Task 2 Tests", "started")
        task2_success = run_task2_tests()
        if task2_success:
            log_ci_step("Task 2 Tests", "completed")
        else:
            log_ci_step("Task 2 Tests", "failed")
            success = False
        
        log_ci_step("Task 3 Tests", "started")
        task3_success = run_task3_tests()
        if task3_success:
            log_ci_step("Task 3 Tests", "completed")
        else:
            log_ci_step("Task 3 Tests", "failed")
            success = False
        
        log_ci_step("Task 4 Tests", "started")
        task4_success = run_task4_tests()
        if task4_success:
            log_ci_step("Task 4 Tests", "completed")
        else:
            log_ci_step("Task 4 Tests", "failed")
            success = False
        
        log_ci_step("Test Execution", "completed" if success else "failed")
        
        # Generate CI report
        log_ci_step("Report Generation", "started")
        report = generate_ci_report()
        log_ci_step("Report Generation", "completed", f"Report saved to logs/ci_report.json")
        
        log_ci_step("CI/CD Test Pipeline", "completed" if success else "failed")
        return success
        
    except Exception as e:
        print(f"CI/CD test execution failed: {e}")
        return False


def main():
    """Main test runner function with validation"""
    parser = argparse.ArgumentParser(description="Run MLOps project validation and tests")
    parser.add_argument("--task1", action="store_true", help="Run only Task 1 tests")
    parser.add_argument("--task2", action="store_true", help="Run only Task 2 tests")
    parser.add_argument("--task3", action="store_true", help="Run only Task 3 tests")
    parser.add_argument("--task4", action="store_true", help="Run only Task 4 tests")
    parser.add_argument("--validate-only", action="store_true", help="Run only environment validation")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation, run tests only")
    parser.add_argument("--pytest", action="store_true", help="Run with pytest")
    parser.add_argument("--ci", action="store_true", help="Run in CI/CD mode with enhanced logging and reporting")
    
    args = parser.parse_args()
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    print("MLOps Heart Disease Project - Test Runner & Validator")
    print("=" * 70)
    
    # Handle CI mode first
    if args.ci:
        print("Running in CI/CD mode...")
        success = run_ci_tests()
        return 0 if success else 1
    
    # Run validation unless skipped
    validation_passed = True
    if not args.skip_validation:
        validation_passed = validate_environment()
        
        if args.validate_only:
            return 0 if validation_passed else 1
        
        if not validation_passed:
            print("\nValidation failed. Fix issues before running tests.")
            print("Use --skip-validation to run tests anyway (not recommended).")
            return 1
    
    # Run tests
    success = True
    
    if args.pytest:
        success = run_pytest()
    elif args.task1:
        success = run_task1_tests()
    elif args.task2:
        success = run_task2_tests()
    elif args.task3:
        success = run_task3_tests()
    elif args.task4:
        success = run_task4_tests()
    else:
        # Run all tests
        print("\n" + "=" * 70)
        print("RUNNING ALL TESTS")
        print("=" * 70)
        
        task1_success = run_task1_tests()
        print("\n")
        task2_success = run_task2_tests()
        print("\n")
        task3_success = run_task3_tests()
        print("\n")
        task4_success = run_task4_tests()
        
        success = task1_success and task2_success and task3_success and task4_success
        
        print("\n" + "=" * 70)
        if success:
            print("ALL TESTS PASSED! Your implementation is working correctly.")
            print("\nNext steps:")
            print("- Review PIPELINE_WORKFLOW.md for team workflows")
            print("- Access Railway MLflow: https://mlflow-tracking-production-53fb.up.railway.app")
            print("- Start developing with: python src/experiment_tracking.py")
        else:
            print("Some tests failed. Please check the output above.")
        print("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)