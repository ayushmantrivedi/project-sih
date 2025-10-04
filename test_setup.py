#!/usr/bin/env python3
"""
Quick test to verify the HealthBot setup is working
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    try:
        from config import get_config, validate_config
        from utils import get_logger
        from chatbot import generate_bot_response
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    try:
        from config import get_config, validate_config
        config = get_config()
        print(f"✅ Configuration loaded - Debug: {config.app.DEBUG}")
        print(f"✅ Model: {config.ml.MODEL_NAME}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_chatbot():
    """Test chatbot functionality"""
    try:
        from chatbot import generate_bot_response
        
        # Test with a simple message
        response = generate_bot_response("Hello")
        print(f"✅ Chatbot response: {response.get('type', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ Chatbot error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing HealthBot Setup...")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Chatbot Test", test_chatbot)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HealthBot is ready to use.")
        print("\n🚀 To start the application, run:")
        print("   python start.py")
        print("\n🐳 Or with Docker:")
        print("   docker-compose up -d")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

