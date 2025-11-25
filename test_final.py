"""
Final test script to verify everything works
"""
import sys
import os
import subprocess

def test_imports():
    """Test all imports"""
    print("🧪 Testing imports...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    modules_to_test = [
        ('src.data_preprocessing', 'generate_synthetic_spe9'),
        ('src.svr_model', 'train_svr'),
        ('src.cnn_lstm_model', 'build_cnn_lstm'),
        ('src.hyperparameter_tuning', 'tune_svr'),
        ('src.utils', 'ensure_dirs')
    ]
    
    all_passed = True
    for module_name, function_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[function_name])
            getattr(module, function_name)
            print(f"✅ {module_name}.{function_name}")
        except ImportError as e:
            print(f"❌ {module_name}.{function_name}: {e}")
            all_passed = False
    
    return all_passed

def test_data_generation():
    """Test data generation"""
    print("\n🧪 Testing data generation...")
    try:
        from src.data_preprocessing import generate_synthetic_spe9, build_feature_table
        df = generate_synthetic_spe9()
        features_df = build_feature_table(df)
        print(f"✅ Data generation: {df.shape} -> {features_df.shape}")
        return True
    except Exception as e:
        print(f"❌ Data generation: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 FINAL PROJECT TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_generation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 ALL TESTS PASSED! Project is ready to run.")
        print("Run: python run_project.py")
    else:
        print("❌ SOME TESTS FAILED. Check the errors above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
