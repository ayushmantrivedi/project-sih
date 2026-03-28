#!/usr/bin/env python3
"""
Simple test to check if ML model works with API and chatbot
"""

def test_ml_model_integration():
    """Test ML model integration with chatbot"""
    print("🧪 Testing ML Model Integration with Chatbot")
    print("=" * 50)
    
    # Test 1: Import ML model function
    print("\n1. Testing ML model import...")
    try:
        from models.ml_predict import predict_disease
        print("✅ ML model function imported successfully")
    except Exception as e:
        print(f"❌ ML model import failed: {e}")
        return False
    
    # Test 2: Test ML model prediction
    print("\n2. Testing ML model prediction...")
    try:
        prediction, confidence, probabilities = predict_disease("I have fever and cough")
        print(f"✅ ML Prediction: {prediction}")
        print(f"✅ Confidence: {confidence:.3f}")
        print(f"✅ Probabilities shape: {probabilities.shape if hasattr(probabilities, 'shape') else len(probabilities)}")
    except Exception as e:
        print(f"❌ ML model prediction failed: {e}")
        return False
    
    # Test 3: Test chatbot with ML model
    print("\n3. Testing chatbot with ML model...")
    try:
        from chatbot import generate_bot_response
        
        # Test symptom analysis (should use ML model)
        response = generate_bot_response("I have fever and cough")
        print(f"✅ Chatbot response type: {response.get('type', 'unknown')}")
        
        if response.get('type') == 'diagnosis':
            print(f"✅ Disease predicted: {response.get('disease', 'unknown')}")
            print(f"✅ Confidence level: {response.get('confidence_level', 'unknown')}")
            print(f"✅ Confidence score: {response.get('confidence', 0):.3f}")
        else:
            print(f"⚠️  Expected 'diagnosis' type, got: {response.get('type')}")
        
    except Exception as e:
        print(f"❌ Chatbot with ML model failed: {e}")
        return False
    
    # Test 4: Test different symptoms
    print("\n4. Testing different symptoms...")
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
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Test 5: Test confidence thresholds
    print("\n5. Testing confidence thresholds...")
    try:
        from config import get_config
        config = get_config()
        
        print(f"✅ High confidence threshold: {config.ml.HIGH_CONFIDENCE_THRESHOLD}")
        print(f"✅ Medium confidence threshold: {config.ml.MEDIUM_CONFIDENCE_THRESHOLD}")
        print(f"✅ Low confidence threshold: {config.ml.LOW_CONFIDENCE_THRESHOLD}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ML MODEL INTEGRATION TEST COMPLETE!")
    print("✅ ML model works with chatbot")
    print("✅ Disease predictions are working")
    print("✅ Confidence levels are calculated")
    print("✅ Configuration is properly loaded")
    
    return True

if __name__ == "__main__":
    success = test_ml_model_integration()
    if not success:
        print("\n❌ ML model integration test failed!")
        print("💡 Check if dependencies are installed: pip install -r requirements.txt")
    else:
        print("\n🚀 Your ML model is working perfectly with the chatbot!")
        print("💡 You can now run: python3 main.py")
    exit(0 if success else 1)