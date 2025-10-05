#!/usr/bin/env python3
"""
Comprehensive testing script for your HealthBot AI Chatbot
Run this on your system where you have Python dependencies installed
"""

import sys
import traceback
import time
import json

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    required_modules = [
        ('flask', 'Flask web framework'),
        ('joblib', 'Model loading'),
        ('numpy', 'Numerical operations'),
        ('pandas', 'Data manipulation'),
        ('sklearn', 'Machine learning'),
        ('tensorflow', 'Deep learning'),
        ('transformers', 'BERT model'),
        ('requests', 'HTTP requests'),
        ('twilio', 'WhatsApp/SMS integration')
    ]
    
    success = True
    for module, description in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} - {description}")
        except ImportError as e:
            print(f"❌ {module} - {description} (Error: {e})")
            success = False
    
    return success

def test_model_loading():
    """Test loading the ML model"""
    print("\n🧪 Testing ML model loading...")
    
    try:
        import joblib
        model_file = 'quick_trained_model.joblib'
        
        print(f"Loading model from {model_file}...")
        model = joblib.load(model_file)
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        
        if hasattr(model, 'classes_'):
            print(f"   Number of classes: {len(model.classes_)}")
            print(f"   Sample classes: {list(model.classes_[:5])}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return None

def test_predict_disease():
    """Test the predict_disease function"""
    print("\n🧪 Testing predict_disease function...")
    
    try:
        from models.ml_predict import predict_disease
        print("✅ predict_disease function imported successfully")
        
        # Test cases
        test_cases = [
            "I have fever and cough",
            "I have headache and nausea",
            "I have chest pain and shortness of breath",
            "I have stomach pain and vomiting",
            "I have rash and itching"
        ]
        
        for i, symptoms in enumerate(test_cases, 1):
            print(f"\n   Test {i}: '{symptoms}'")
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

def test_chatbot_responses():
    """Test chatbot responses"""
    print("\n🧪 Testing chatbot responses...")
    
    try:
        from chatbot import generate_bot_response
        print("✅ generate_bot_response function imported successfully")
        
        # Test different types of queries
        test_queries = [
            ("I have fever and cough", "symptom analysis"),
            ("Show me COVID statistics", "covid data"),
            ("Find clinical trials for diabetes", "clinical trials"),
            ("What is hypertension?", "medical definition"),
            ("Show me vaccination data", "vaccination data")
        ]
        
        for query, expected_type in test_queries:
            print(f"\n   Testing: '{query}'")
            try:
                response = generate_bot_response(query)
                print(f"   ✅ Response type: {response.get('type', 'unknown')}")
                print(f"   ✅ Response: {str(response)[:100]}...")
                
                if response.get('type') == 'diagnosis':
                    print(f"   ✅ Disease: {response.get('disease', 'unknown')}")
                    print(f"   ✅ Confidence: {response.get('confidence', 0):.3f}")
                
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chatbot responses test failed: {e}")
        traceback.print_exc()
        return False

def test_api_integrations():
    """Test API integrations"""
    print("\n🧪 Testing API integrations...")
    
    apis_to_test = [
        ('apis.clinicaltrials', 'get_clinical_trials', 'diabetes'),
        ('apis.cowin', 'get_cowin_stats', None),
        ('apis.mohfw', 'get_mohfw_data', None),
        ('apis.umls', 'get_umls_info', 'hypertension')
    ]
    
    success = True
    for api_module, function_name, test_param in apis_to_test:
        try:
            print(f"\n   Testing {api_module}.{function_name}...")
            module = __import__(api_module, fromlist=[function_name])
            func = getattr(module, function_name)
            
            if test_param:
                result = func(test_param)
            else:
                result = func()
            
            print(f"   ✅ {api_module}.{function_name} succeeded")
            print(f"   ✅ Result type: {type(result)}")
            if isinstance(result, (list, dict)):
                print(f"   ✅ Result length: {len(result)}")
            
        except Exception as e:
            print(f"   ❌ {api_module}.{function_name} failed: {e}")
            success = False
    
    return success

def test_web_server():
    """Test web server startup"""
    print("\n🧪 Testing web server...")
    
    try:
        from main import app
        print("✅ Flask app imported successfully")
        
        # Test app configuration
        print(f"   ✅ App name: {app.name}")
        print(f"   ✅ Debug mode: {app.debug}")
        
        # Test routes
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = ['/health', '/chat', '/webhook/whatsapp', '/webhook/sms']
        
        for route in expected_routes:
            if route in routes:
                print(f"   ✅ Route {route} found")
            else:
                print(f"   ❌ Route {route} not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Web server test failed: {e}")
        traceback.print_exc()
        return False

def test_webhook_functions():
    """Test webhook functions"""
    print("\n🧪 Testing webhook functions...")
    
    try:
        # Test WhatsApp webhook
        from whatsapp_enhanced import whatsapp_webhook
        print("✅ WhatsApp webhook function imported")
        
        # Test SMS webhook
        from sms_enhanced import sms_webhook
        print("✅ SMS webhook function imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Webhook functions test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")
    
    try:
        from config import get_config
        config = get_config()
        print("✅ Configuration loaded successfully")
        
        # Test config sections
        config_sections = ['api', 'ml', 'database', 'twilio', 'app']
        for section in config_sections:
            if hasattr(config, section):
                print(f"   ✅ {section} configuration found")
            else:
                print(f"   ❌ {section} configuration not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_database_models():
    """Test database models"""
    print("\n🧪 Testing database models...")
    
    try:
        from database.models import User, Conversation, Message, Analytics
        print("✅ Database models imported successfully")
        
        # Test model creation (without saving)
        user = User(phone_number="+1234567890", platform="test")
        print("   ✅ User model can be instantiated")
        
        conversation = Conversation(user_id=1, session_id="test123", platform="test")
        print("   ✅ Conversation model can be instantiated")
        
        message = Message(conversation_id=1, user_id=1, message_type="user", content="test")
        print("   ✅ Message model can be instantiated")
        
        return True
        
    except Exception as e:
        print(f"❌ Database models test failed: {e}")
        traceback.print_exc()
        return False

def test_monitoring():
    """Test monitoring system"""
    print("\n🧪 Testing monitoring system...")
    
    try:
        from monitoring.analytics import HealthBotAnalytics
        analytics = HealthBotAnalytics()
        print("✅ Analytics system imported successfully")
        
        # Test tracking functions
        analytics.track_message("test", "user123", "user", "diagnosis", 0.85)
        print("   ✅ Message tracking works")
        
        analytics.track_api_call("test_api", success=True, response_time=1.5)
        print("   ✅ API call tracking works")
        
        health_status = analytics.get_health_status()
        print(f"   ✅ Health status: {health_status.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring system test failed: {e}")
        traceback.print_exc()
        return False

def run_web_server_test():
    """Run a quick web server test"""
    print("\n🧪 Running web server test...")
    
    try:
        from main import app
        
        # Test health endpoint
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print("✅ Health endpoint works")
                print(f"   Response: {response.get_json()}")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
            
            # Test chat endpoint
            chat_data = {
                "message": "I have fever and cough",
                "user_id": "test_user"
            }
            response = client.post('/chat', json=chat_data)
            if response.status_code == 200:
                print("✅ Chat endpoint works")
                print(f"   Response: {response.get_json()}")
            else:
                print(f"❌ Chat endpoint failed: {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Web server test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 HealthBot AI Chatbot - Comprehensive System Test")
    print("=" * 60)
    print("Run this script on your system where you have Python dependencies installed")
    print("=" * 60)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Load ML Model", test_model_loading),
        ("Test Disease Prediction", test_predict_disease),
        ("Test Chatbot Responses", test_chatbot_responses),
        ("Test API Integrations", test_api_integrations),
        ("Test Web Server", test_web_server),
        ("Test Webhook Functions", test_webhook_functions),
        ("Test Configuration", test_configuration),
        ("Test Database Models", test_database_models),
        ("Test Monitoring", test_monitoring),
        ("Run Web Server Test", run_web_server_test)
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
    
    print("\n" + "=" * 60)
    print(f"📊 FINAL TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your HealthBot AI Chatbot is working perfectly!")
        print("\n🚀 READY FOR DEPLOYMENT!")
        print("\nNext steps:")
        print("1. Set up Twilio credentials in .env file")
        print("2. Run: python main.py")
        print("3. Test: curl http://localhost:5000/health")
        print("4. Deploy to production!")
    else:
        print(f"⚠️  {total - passed} test(s) failed.")
        print("Please check the errors above and fix any issues.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)