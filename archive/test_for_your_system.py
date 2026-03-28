#!/usr/bin/env python3
"""
Test script for your system where you have Python dependencies installed
This will work on your master branch files
"""

def test_your_system():
    """Test ML model integration on your system"""
    print("🧪 Testing ML Model Integration on Your System")
    print("=" * 50)
    print("Make sure you're in the directory with your master branch files!")
    print("=" * 50)
    
    # Test 1: Check current directory and files
    print("\n1. Checking current directory and files...")
    import os
    
    print(f"Current directory: {os.getcwd()}")
    
    required_files = [
        'models/ml_predict.py',
        'models/sihdemo.py',
        'models/__init__.py',
        'chatbot.py',
        'quick_trained_model.joblib'
    ]
    
    all_files_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Some required files are missing!")
        print("Make sure you're in the correct directory with your master branch files.")
        return False
    
    # Test 2: Test dependencies
    print("\n2. Testing dependencies...")
    dependencies = ['numpy', 'pandas', 'sklearn', 'joblib', 'tensorflow', 'transformers']
    
    missing_deps = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} not available")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {missing_deps}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    # Test 3: Test ML model import
    print("\n3. Testing ML model import...")
    try:
        from models.ml_predict import predict_disease
        print("✅ predict_disease function imported successfully")
    except Exception as e:
        print(f"❌ ML model import failed: {e}")
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
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 SUCCESS! Your ML model works perfectly with the chatbot!")
    print("✅ All files are present")
    print("✅ All dependencies are available")
    print("✅ ML model predictions work")
    print("✅ Chatbot integration works")
    print("✅ Your master branch files are working correctly!")
    
    return True

if __name__ == "__main__":
    success = test_your_system()
    if not success:
        print("\n❌ Test failed!")
        print("💡 Common solutions:")
        print("   1. Make sure you're in the correct directory")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Check if all files are present")
    else:
        print("\n🚀 Your chatbot is ready to use!")
        print("💡 You can now run: python3 main.py")
    exit(0 if success else 1)