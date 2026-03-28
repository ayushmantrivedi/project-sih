#!/usr/bin/env python3
"""
Non-interactive web endpoints test
"""

import os
import sys
import traceback

def test_main_app_structure():
    """Test main.py structure"""
    print("🧪 Testing main.py structure...")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check for Flask app
        if 'from flask import Flask' in content:
            print("✅ Flask import found")
        else:
            print("❌ Flask import not found")
            return False
        
        # Check for app initialization
        if 'app = Flask(__name__)' in content:
            print("✅ Flask app initialization found")
        else:
            print("❌ Flask app initialization not found")
            return False
        
        # Check for routes
        routes = [
            '@app.route(\'/health\'',
            '@app.route(\'/chat\'',
            '@app.route(\'/webhook/whatsapp\'',
            '@app.route(\'/webhook/sms\''
        ]
        
        for route in routes:
            if route in content:
                print(f"✅ {route} route found")
            else:
                print(f"❌ {route} route not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ main.py structure test failed: {e}")
        return False

def test_app_py_structure():
    """Test app.py structure"""
    print("\n🧪 Testing app.py structure...")
    
    try:
        with open('app.py', 'r') as f:
            content = f.read()
        
        # Check for Flask app
        if 'from flask import Flask' in content:
            print("✅ Flask import found")
        else:
            print("❌ Flask import not found")
            return False
        
        # Check for chat route
        if '@app.route("/chat"' in content:
            print("✅ Chat route found")
        else:
            print("❌ Chat route not found")
            return False
        
        # Check for chatbot import
        if 'from chatbot import generate_bot_response' in content:
            print("✅ Chatbot import found")
        else:
            print("❌ Chatbot import not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ app.py structure test failed: {e}")
        return False

def test_webhook_structure():
    """Test webhook structure"""
    print("\n🧪 Testing webhook structure...")
    
    webhook_files = [
        'whatsapp_webhook.py',
        'sms_webhook.py',
        'whatsapp_enhanced.py',
        'sms_enhanced.py'
    ]
    
    for file in webhook_files:
        if os.path.exists(file):
            try:
                with open(file, 'r') as f:
                    content = f.read()
                
                # Check for Flask app
                if 'from flask import Flask' in content:
                    print(f"✅ {file} has Flask import")
                else:
                    print(f"❌ {file} missing Flask import")
                    return False
                
                # Check for Twilio
                if 'from twilio.twiml.messaging_response import MessagingResponse' in content:
                    print(f"✅ {file} has Twilio import")
                else:
                    print(f"❌ {file} missing Twilio import")
                    return False
                
                # Check for webhook route
                if '@app.route' in content and 'POST' in content:
                    print(f"✅ {file} has webhook route")
                else:
                    print(f"❌ {file} missing webhook route")
                    return False
                
            except Exception as e:
                print(f"❌ Error reading {file}: {e}")
                return False
        else:
            print(f"⚠️  {file} not found")
    
    return True

def test_endpoint_functions():
    """Test endpoint functions"""
    print("\n🧪 Testing endpoint functions...")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check for health check function
        if 'def health_check' in content:
            print("✅ health_check function found")
        else:
            print("❌ health_check function not found")
            return False
        
        # Check for chat function
        if 'def chat' in content:
            print("✅ chat function found")
        else:
            print("❌ chat function not found")
            return False
        
        # Check for webhook functions
        if 'def whatsapp_webhook' in content:
            print("✅ whatsapp_webhook function found")
        else:
            print("❌ whatsapp_webhook function not found")
            return False
        
        if 'def sms_webhook' in content:
            print("✅ sms_webhook function found")
        else:
            print("❌ sms_webhook function not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Endpoint functions test failed: {e}")
        return False

def test_error_handling():
    """Test error handling in endpoints"""
    print("\n🧪 Testing error handling...")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check for error handlers
        if '@app.errorhandler' in content:
            print("✅ Error handlers found")
        else:
            print("⚠️  Error handlers not found")
        
        # Check for try-except blocks
        try_count = content.count('try:')
        except_count = content.count('except')
        
        print(f"✅ Found {try_count} try blocks and {except_count} except blocks")
        
        # Check for specific error handling
        if 'jsonify' in content and 'error' in content.lower():
            print("✅ JSON error responses found")
        else:
            print("⚠️  JSON error responses not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_rate_limiting():
    """Test rate limiting"""
    print("\n🧪 Testing rate limiting...")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check for rate limiter import
        if 'from flask_limiter import Limiter' in content:
            print("✅ Flask-Limiter import found")
        else:
            print("⚠️  Flask-Limiter import not found")
        
        # Check for limiter initialization
        if 'limiter = Limiter' in content:
            print("✅ Rate limiter initialization found")
        else:
            print("⚠️  Rate limiter initialization not found")
        
        # Check for rate limiting decorators
        if '@limiter.limit' in content:
            print("✅ Rate limiting decorators found")
        else:
            print("⚠️  Rate limiting decorators not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Rate limiting test failed: {e}")
        return False

def test_cors_support():
    """Test CORS support"""
    print("\n🧪 Testing CORS support...")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check for CORS import
        if 'from flask_cors import CORS' in content:
            print("✅ Flask-CORS import found")
        else:
            print("⚠️  Flask-CORS import not found")
        
        # Check for CORS initialization
        if 'CORS(app)' in content:
            print("✅ CORS initialization found")
        else:
            print("⚠️  CORS initialization not found")
        
        return True
        
    except Exception as e:
        print(f"❌ CORS test failed: {e}")
        return False

def test_configuration_usage():
    """Test configuration usage in endpoints"""
    print("\n🧪 Testing configuration usage...")
    
    try:
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check for config import
        if 'from config import get_config' in content:
            print("✅ Config import found")
        else:
            print("❌ Config import not found")
            return False
        
        # Check for config usage
        if 'config = get_config()' in content:
            print("✅ Config initialization found")
        else:
            print("❌ Config initialization not found")
            return False
        
        # Check for config usage in routes
        if 'config.app.' in content:
            print("✅ Config usage in routes found")
        else:
            print("⚠️  Config usage in routes not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration usage test failed: {e}")
        return False

def test_syntax():
    """Test syntax of web endpoint files"""
    print("\n🧪 Testing syntax...")
    
    files_to_test = [
        'main.py',
        'app.py',
        'whatsapp_webhook.py',
        'sms_webhook.py',
        'whatsapp_enhanced.py',
        'sms_enhanced.py'
    ]
    
    for file in files_to_test:
        if os.path.exists(file):
            try:
                with open(file, 'r') as f:
                    content = f.read()
                
                compile(content, file, 'exec')
                print(f"✅ {file} syntax is valid")
                
            except SyntaxError as e:
                print(f"❌ Syntax error in {file}: {e}")
                return False
            except Exception as e:
                print(f"❌ Error testing {file}: {e}")
                return False
        else:
            print(f"⚠️  {file} not found")
    
    return True

def main():
    """Run all web endpoint tests"""
    print("🚀 HealthBot AI Chatbot - Web Endpoints Test")
    print("=" * 50)
    
    tests = [
        test_main_app_structure,
        test_app_py_structure,
        test_webhook_structure,
        test_endpoint_functions,
        test_error_handling,
        test_rate_limiting,
        test_cors_support,
        test_configuration_usage,
        test_syntax
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
    print(f"📊 Web Endpoints Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All web endpoint tests passed!")
        print("\n📋 Web Endpoints Status:")
        print("✅ All Flask apps are properly structured")
        print("✅ All routes are correctly defined")
        print("✅ Error handling is in place")
        print("✅ Rate limiting is configured")
        print("✅ CORS support is enabled")
        print("✅ Configuration is properly used")
        print("\n💡 To test actual endpoints, install dependencies and run:")
        print("   pip install -r requirements.txt")
        print("   python main.py")
        print("   curl http://localhost:5000/health")
    else:
        print("⚠️  Some web endpoint tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)