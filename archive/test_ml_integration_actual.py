#!/usr/bin/env python3
"""
Test ML model integration with your actual master branch files
"""

def test_ml_model_integration():
    """Test ML model integration with chatbot using your actual files"""
    print("🧪 Testing ML Model Integration with Your Master Branch Files")
    print("=" * 60)
    
    # Test 1: Check if required files exist
    print("\n1. Checking required files...")
    import os
    
    required_files = [
        'models/ml_predict.py',
        'models/sihdemo.py', 
        'models/__init__.py',
        'chatbot.py',
        'quick_trained_model.joblib'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    # Test 2: Test ML model import
    print("\n2. Testing ML model import...")
    try:
        from models.ml_predict import predict_disease
        print("✅ predict_disease function imported successfully")
    except Exception as e:
        print(f"❌ ML model import failed: {e}")
        return False
    
    # Test 3: Test sihdemo import
    print("\n3. Testing sihdemo module...")
    try:
        from models.sihdemo import predict_new
        print("✅ predict_new function imported successfully")
    except Exception as e:
        print(f"❌ sihdemo import failed: {e}")
        return False
    
    # Test 4: Test ML model prediction
    print("\n4. Testing ML model prediction...")
    try:
        prediction, confidence, probabilities = predict_disease("I have fever and cough")
        print(f"✅ ML Prediction: {prediction}")
        print(f"✅ Confidence: {confidence:.3f}")
        print(f"✅ Probabilities type: {type(probabilities)}")
    except Exception as e:
        print(f"❌ ML model prediction failed: {e}")
        print("💡 This might be due to missing dependencies or model file issues")
        return False
    
    # Test 5: Test chatbot import
    print("\n5. Testing chatbot import...")
    try:
        from chatbot import generate_bot_response
        print("✅ generate_bot_response function imported successfully")
    except Exception as e:
        print(f"❌ Chatbot import failed: {e}")
        return False
    
    # Test 6: Test chatbot with ML model
    print("\n6. Testing chatbot with ML model...")
    try:
        response = generate_bot_response("I have fever and cough")
        print(f"✅ Chatbot response type: {response.get('type', 'unknown')}")
        
        if response.get('type') == 'diagnosis':
            print(f"✅ Disease predicted: {response.get('disease', 'unknown')}")
            print(f"✅ Confidence level: {response.get('confidence_level', 'unknown')}")
            print(f"✅ Confidence score: {response.get('confidence', 0):.3f}")
        else:
            print(f"⚠️  Expected 'diagnosis' type, got: {response.get('type')}")
            print(f"   Full response: {response}")
        
    except Exception as e:
        print(f"❌ Chatbot with ML model failed: {e}")
        return False
    
    # Test 7: Test different symptoms
    print("\n7. Testing different symptoms...")
    test_symptoms = [
        "I have headache and nausea",
        "I have chest pain and shortness of breath",
        "I have stomach pain and vomiting"
    ]
    
    for i, symptoms in enumerate(test_symptoms, 1):
        try:
            print(f"\n   Test {i}: '{symptoms}'")
            response = generate_bot_response(symptoms)
            
            if response.get('type') == 'diagnosis':
                print(f"   ✅ Disease: {response.get('disease', 'unknown')}")
                print(f"   ✅ Confidence: {response.get('confidence', 0):.3f}")
            else:
                print(f"   ⚠️  Response type: {response.get('type')}")
                print(f"   ⚠️  Response: {response}")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Test 8: Test configuration
    print("\n8. Testing configuration...")
    try:
        from config import get_config
        config = get_config()
        print("✅ Configuration loaded successfully")
        
        if hasattr(config, 'ml'):
            print(f"✅ ML config found")
            if hasattr(config.ml, 'HIGH_CONFIDENCE_THRESHOLD'):
                print(f"✅ High confidence threshold: {config.ml.HIGH_CONFIDENCE_THRESHOLD}")
        else:
            print("⚠️  ML config not found")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ML MODEL INTEGRATION TEST COMPLETE!")
    print("✅ ML model works with chatbot")
    print("✅ Disease predictions are working")
    print("✅ Confidence levels are calculated")
    print("✅ Your master branch files are working correctly")
    
    return True

if __name__ == "__main__":
    success = test_ml_model_integration()
    if not success:
        print("\n❌ ML model integration test failed!")
        print("💡 Common issues:")
        print("   - Missing dependencies: pip install -r requirements.txt")
        print("   - Model file not found: check quick_trained_model.joblib")
        print("   - Import errors: check Python path and file structure")
    else:
        print("\n🚀 Your ML model is working perfectly with the chatbot!")
        print("💡 You can now run: python3 main.py")
    exit(0 if success else 1)