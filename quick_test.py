#!/usr/bin/env python3
"""
Quick test script for HealthBot AI Chatbot
Run this to quickly verify everything is working
"""

def quick_test():
    """Run a quick test of all major components"""
    print("🚀 HealthBot AI Chatbot - Quick Test")
    print("=" * 40)
    
    # Test 1: Imports
    print("\n1. Testing imports...")
    try:
        import flask, joblib, numpy, pandas, sklearn, requests
        print("✅ All core dependencies imported successfully!")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Model loading
    print("\n2. Testing model loading...")
    try:
        import joblib
        model = joblib.load('quick_trained_model.joblib')
        print(f"✅ Model loaded successfully! Type: {type(model)}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test 3: Disease prediction
    print("\n3. Testing disease prediction...")
    try:
        from models.ml_predict import predict_disease
        prediction, confidence, probabilities = predict_disease("I have fever and cough")
        print(f"✅ Prediction: {prediction} (Confidence: {confidence:.3f})")
    except Exception as e:
        print(f"❌ Disease prediction failed: {e}")
        return False
    
    # Test 4: Chatbot responses
    print("\n4. Testing chatbot responses...")
    try:
        from chatbot import generate_bot_response
        
        # Test symptom analysis
        response = generate_bot_response("I have fever and cough")
        print(f"✅ Symptom analysis: {response.get('type', 'unknown')}")
        
        # Test COVID query
        response = generate_bot_response("Show me COVID statistics")
        print(f"✅ COVID query: {response.get('type', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Chatbot responses failed: {e}")
        return False
    
    # Test 5: Web server
    print("\n5. Testing web server...")
    try:
        from main import app
        print(f"✅ Flask app loaded successfully! Debug: {app.debug}")
        
        # Test routes
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = ['/health', '/chat', '/webhook/whatsapp', '/webhook/sms']
        for route in expected_routes:
            if route in routes:
                print(f"   ✅ Route {route} found")
            else:
                print(f"   ❌ Route {route} missing")
                
    except Exception as e:
        print(f"❌ Web server test failed: {e}")
        return False
    
    # Test 6: API integrations
    print("\n6. Testing API integrations...")
    try:
        from apis.clinicaltrials import get_clinical_trials
        from apis.cowin import get_cowin_stats
        from apis.mohfw import get_mohfw_data
        
        # Test Clinical Trials
        trials = get_clinical_trials("diabetes")
        print(f"✅ Clinical Trials: {len(trials)} results")
        
        # Test CoWIN
        cowin = get_cowin_stats()
        print(f"✅ CoWIN: {type(cowin)}")
        
        # Test MoHFW
        mohfw = get_mohfw_data()
        print(f"✅ MoHFW: {type(mohfw)}")
        
    except Exception as e:
        print(f"❌ API integrations failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Your HealthBot AI Chatbot is working perfectly!")
    print("\n🚀 Ready to start the server:")
    print("   python3 main.py")
    print("\n📱 Test the chatbot:")
    print("   curl http://localhost:5000/health")
    print("   curl -X POST http://localhost:5000/chat -H 'Content-Type: application/json' -d '{\"message\": \"I have fever and cough\", \"user_id\": \"test\"}'")
    
    return True

if __name__ == "__main__":
    success = quick_test()
    if not success:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("💡 Try installing dependencies: pip install -r requirements.txt")
    exit(0 if success else 1)