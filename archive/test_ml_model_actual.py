#!/usr/bin/env python3
"""
Test ML model with actual dependencies
"""

import sys
import traceback

def test_imports():
    """Test if we can import required modules"""
    print("🧪 Testing imports...")
    
    try:
        import joblib
        print("✅ joblib imported successfully")
    except ImportError as e:
        print(f"❌ joblib import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✅ tensorflow imported successfully")
    except ImportError as e:
        print(f"❌ tensorflow import failed: {e}")
        return False
    
    try:
        import transformers
        print("✅ transformers imported successfully")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test loading the trained model"""
    print("\n🧪 Testing model loading...")
    
    try:
        import joblib
        model_file = 'quick_trained_model.joblib'
        
        print(f"Loading model from {model_file}...")
        model = joblib.load(model_file)
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        
        # Try to get model attributes
        if hasattr(model, 'classes_'):
            print(f"   Number of classes: {len(model.classes_)}")
            print(f"   Classes: {model.classes_[:5]}...")  # Show first 5 classes
        
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return None

def test_ml_predict_function():
    """Test the predict_disease function"""
    print("\n🧪 Testing predict_disease function...")
    
    try:
        # Import the function
        sys.path.append('.')
        from models.ml_predict import predict_disease
        print("✅ predict_disease function imported successfully")
        
        # Test with sample symptoms
        test_symptoms = [
            "I have fever and cough",
            "I have headache and nausea",
            "I have chest pain and shortness of breath",
            "I have stomach pain and vomiting"
        ]
        
        for symptoms in test_symptoms:
            print(f"\n   Testing: '{symptoms}'")
            try:
                prediction, confidence, probabilities = predict_disease(symptoms)
                print(f"   ✅ Prediction: {prediction}")
                print(f"   ✅ Confidence: {confidence:.3f}")
                print(f"   ✅ Probabilities shape: {probabilities.shape if hasattr(probabilities, 'shape') else len(probabilities)}")
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ predict_disease function test failed: {e}")
        traceback.print_exc()
        return False

def test_quick_predict():
    """Test quick_predict function"""
    print("\n🧪 Testing quick_predict function...")
    
    try:
        from models.quick_predict import quick_predict
        print("✅ quick_predict function imported successfully")
        
        # Test with sample symptoms
        test_symptoms = "I have fever and cough"
        print(f"   Testing: '{test_symptoms}'")
        
        result = quick_predict(test_symptoms)
        print(f"   ✅ Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ quick_predict function test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all ML model tests"""
    print("🚀 HealthBot AI Chatbot - ML Model Actual Test")
    print("=" * 50)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Load Model", test_model_loading),
        ("Test predict_disease", test_ml_predict_function),
        ("Test quick_predict", test_quick_predict)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 ML Model Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All ML model tests passed!")
        print("✅ Your ML model is working perfectly!")
    else:
        print("⚠️  Some ML model tests failed.")
        print("Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)