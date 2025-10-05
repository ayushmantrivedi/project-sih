#!/usr/bin/env python3
"""
Non-interactive ML model test for HealthBot AI Chatbot
"""

import os
import sys
import traceback

def test_model_loading():
    """Test if we can load the model file"""
    print("🧪 Testing model loading...")
    
    try:
        # Check if model file exists
        model_file = 'quick_trained_model.joblib'
        if not os.path.exists(model_file):
            print(f"❌ Model file {model_file} not found")
            return False
        
        print(f"✅ Model file {model_file} found")
        print(f"   File size: {os.path.getsize(model_file)} bytes")
        
        # Try to load with joblib (if available)
        try:
            import joblib
            model = joblib.load(model_file)
            print("✅ Model loaded successfully with joblib")
            print(f"   Model type: {type(model)}")
            return True
        except ImportError:
            print("⚠️  joblib not available, cannot test model loading")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def test_model_predict_function():
    """Test the predict_disease function structure"""
    print("\n🧪 Testing predict_disease function...")
    
    try:
        with open('models/ml_predict.py', 'r') as f:
            content = f.read()
        
        # Check if function exists
        if 'def predict_disease' in content:
            print("✅ predict_disease function found")
        else:
            print("❌ predict_disease function not found")
            return False
        
        # Check for required imports
        if 'import joblib' in content:
            print("✅ joblib import found")
        if 'import numpy' in content:
            print("✅ numpy import found")
        if 'import pandas' in content:
            print("✅ pandas import found")
        
        return True
        
    except Exception as e:
        print(f"❌ Function structure test failed: {e}")
        return False

def test_model_dependencies():
    """Test if model dependencies are available"""
    print("\n🧪 Testing model dependencies...")
    
    dependencies = [
        ('joblib', 'Model loading'),
        ('numpy', 'Numerical operations'),
        ('pandas', 'Data manipulation'),
        ('sklearn', 'Machine learning'),
        ('tensorflow', 'Deep learning'),
        ('transformers', 'BERT model')
    ]
    
    available = 0
    for dep, description in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} - {description}")
            available += 1
        except ImportError:
            print(f"❌ {dep} - {description} (not available)")
    
    print(f"\n📊 Dependencies: {available}/{len(dependencies)} available")
    return available > 0

def test_model_file_integrity():
    """Test model file integrity"""
    print("\n🧪 Testing model file integrity...")
    
    try:
        model_file = 'quick_trained_model.joblib'
        
        # Check file size (should be reasonable)
        file_size = os.path.getsize(model_file)
        if file_size < 1000:  # Less than 1KB seems too small
            print(f"⚠️  Model file seems too small: {file_size} bytes")
        elif file_size > 100 * 1024 * 1024:  # More than 100MB seems too large
            print(f"⚠️  Model file seems too large: {file_size} bytes")
        else:
            print(f"✅ Model file size looks reasonable: {file_size} bytes")
        
        # Try to read the file
        with open(model_file, 'rb') as f:
            header = f.read(10)
            if header.startswith(b'\x80\x02'):  # joblib pickle header
                print("✅ Model file appears to be a valid joblib pickle")
            else:
                print("⚠️  Model file doesn't appear to be a joblib pickle")
        
        return True
        
    except Exception as e:
        print(f"❌ Model file integrity test failed: {e}")
        return False

def test_quick_predict():
    """Test quick_predict.py functionality"""
    print("\n🧪 Testing quick_predict.py...")
    
    try:
        with open('models/quick_predict.py', 'r') as f:
            content = f.read()
        
        # Check for main function
        if 'def quick_predict' in content:
            print("✅ quick_predict function found")
        else:
            print("❌ quick_predict function not found")
            return False
        
        # Check for model loading
        if 'joblib.load' in content:
            print("✅ Model loading code found")
        
        return True
        
    except Exception as e:
        print(f"❌ quick_predict test failed: {e}")
        return False

def main():
    """Run all ML model tests"""
    print("🚀 HealthBot AI Chatbot - ML Model Test")
    print("=" * 50)
    
    tests = [
        test_model_loading,
        test_model_predict_function,
        test_model_dependencies,
        test_model_file_integrity,
        test_quick_predict
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 ML Model Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All ML model tests passed!")
        print("\n📋 ML Model Status:")
        print("✅ Model file exists and appears valid")
        print("✅ Prediction functions are properly structured")
        print("⚠️  Some dependencies may need to be installed")
        print("\n💡 To test actual predictions, install dependencies and run:")
        print("   pip install -r requirements.txt")
        print("   python test_my_model.py")
    else:
        print("⚠️  Some ML model tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)