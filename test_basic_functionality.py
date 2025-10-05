#!/usr/bin/env python3
"""
Basic functionality test for HealthBot AI Chatbot
This test runs without external dependencies to verify basic structure
"""

import os
import sys
import json
import traceback

def test_file_structure():
    """Test if all required files exist"""
    print("🧪 Testing file structure...")
    
    required_files = [
        'chatbot.py',
        'main.py', 
        'app.py',
        'requirements.txt',
        'quick_trained_model.joblib',
        'config/settings.py',
        'utils/logger.py',
        'apis/clinicaltrials.py',
        'apis/cowin.py',
        'apis/mohfw.py',
        'apis/umls.py',
        'models/ml_predict.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_config_loading():
    """Test configuration loading"""
    print("\n🧪 Testing configuration loading...")
    
    try:
        # Test if we can read the config file
        with open('config/settings.py', 'r') as f:
            config_content = f.read()
        
        print("✅ Config file readable")
        
        # Check if it contains expected classes
        if 'class APIConfig' in config_content:
            print("✅ APIConfig class found")
        if 'class MLConfig' in config_content:
            print("✅ MLConfig class found")
        if 'class TwilioConfig' in config_content:
            print("✅ TwilioConfig class found")
            
        return True
        
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_chatbot_structure():
    """Test chatbot.py structure"""
    print("\n🧪 Testing chatbot structure...")
    
    try:
        with open('chatbot.py', 'r') as f:
            chatbot_content = f.read()
        
        # Check for main function
        if 'def generate_bot_response' in chatbot_content:
            print("✅ generate_bot_response function found")
        else:
            print("❌ generate_bot_response function not found")
            return False
            
        # Check for imports
        if 'from apis.clinicaltrials import get_clinical_trials' in chatbot_content:
            print("✅ Clinical trials import found")
        if 'from apis.cowin import get_cowin_stats' in chatbot_content:
            print("✅ CoWIN import found")
        if 'from models.ml_predict import predict_disease' in chatbot_content:
            print("✅ ML prediction import found")
            
        return True
        
    except Exception as e:
        print(f"❌ Chatbot structure error: {e}")
        return False

def test_api_structure():
    """Test API files structure"""
    print("\n🧪 Testing API structure...")
    
    api_files = [
        'apis/clinicaltrials.py',
        'apis/cowin.py', 
        'apis/mohfw.py',
        'apis/umls.py'
    ]
    
    for api_file in api_files:
        try:
            with open(api_file, 'r') as f:
                content = f.read()
            
            # Check for main function
            if 'def get_' in content:
                print(f"✅ {api_file} has get functions")
            else:
                print(f"❌ {api_file} missing get functions")
                return False
                
        except Exception as e:
            print(f"❌ Error reading {api_file}: {e}")
            return False
    
    return True

def test_model_structure():
    """Test model files structure"""
    print("\n🧪 Testing model structure...")
    
    try:
        with open('models/ml_predict.py', 'r') as f:
            model_content = f.read()
        
        if 'def predict_disease' in model_content:
            print("✅ predict_disease function found")
        else:
            print("❌ predict_disease function not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Model structure error: {e}")
        return False

def test_webhook_structure():
    """Test webhook files structure"""
    print("\n🧪 Testing webhook structure...")
    
    webhook_files = [
        'whatsapp_webhook.py',
        'sms_webhook.py',
        'whatsapp_enhanced.py',
        'sms_enhanced.py'
    ]
    
    for webhook_file in webhook_files:
        if os.path.exists(webhook_file):
            try:
                with open(webhook_file, 'r') as f:
                    content = f.read()
                
                if 'def whatsapp' in content or 'def sms' in content:
                    print(f"✅ {webhook_file} has webhook functions")
                else:
                    print(f"❌ {webhook_file} missing webhook functions")
                    return False
                    
            except Exception as e:
                print(f"❌ Error reading {webhook_file}: {e}")
                return False
        else:
            print(f"⚠️  {webhook_file} not found")
    
    return True

def test_environment_setup():
    """Test environment setup"""
    print("\n🧪 Testing environment setup...")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file found")
        try:
            with open('.env', 'r') as f:
                env_content = f.read()
            
            # Check for required variables
            required_vars = ['TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN']
            for var in required_vars:
                if var in env_content:
                    print(f"✅ {var} found in .env")
                else:
                    print(f"⚠️  {var} not found in .env")
        except Exception as e:
            print(f"❌ Error reading .env: {e}")
    else:
        print("⚠️  .env file not found")
    
    return True

def main():
    """Run all tests"""
    print("🚀 HealthBot AI Chatbot - Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_config_loading,
        test_chatbot_structure,
        test_api_structure,
        test_model_structure,
        test_webhook_structure,
        test_environment_setup
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
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your project structure looks good.")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up Twilio credentials in .env file")
        print("3. Test ML model: python test_my_model.py")
        print("4. Run the app: python main.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)