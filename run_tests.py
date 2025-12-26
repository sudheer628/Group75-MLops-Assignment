"""
Test Runner for MLOps Heart Disease Project
Runs all tests for Task 1 and Task 2

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --task1      # Run only Task 1 tests
    python run_tests.py --task2      # Run only Task 2 tests
    python run_tests.py --pytest    # Run with pytest (requires pytest installed)
"""

import sys
import argparse
from pathlib import Path

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

def run_task3_tests():
    """Run Task 3 tests"""
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

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Run MLOps project tests")
    parser.add_argument("--task1", action="store_true", help="Run only Task 1 tests")
    parser.add_argument("--task2", action="store_true", help="Run only Task 2 tests")
    parser.add_argument("--task3", action="store_true", help="Run only Task 3 tests")
    parser.add_argument("--task4", action="store_true", help="Run only Task 4 tests")
    parser.add_argument("--pytest", action="store_true", help="Run with pytest")
    
    args = parser.parse_args()
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
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
        print("Running All Tests for MLOps Heart Disease Project")
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
        else:
            print("Some tests failed. Please check the output above.")
        print("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)