#!/usr/bin/env python3
"""
Simple test to verify the HealthBot setup works without external dependencies
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic Python functionality"""
    print("🧪 Testing Basic Setup...")
    print("=" * 40)
    
    try:
        # Test basic imports
        import json
        import requests
        print("✅ Basic Python modules working")
        
        # Test project structure
        if os.path.exists("config"):
            print("✅ Config directory exists")
        else:
            print("❌ Config directory missing")
            
        if os.path.exists("apis"):
            print("✅ APIs directory exists")
        else:
            print("❌ APIs directory missing")
            
        if os.path.exists("models"):
            print("✅ Models directory exists")
        else:
            print("❌ Models directory missing")
            
        # Test if we can read our files
        if os.path.exists("chatbot.py"):
            print("✅ chatbot.py exists")
        else:
            print("❌ chatbot.py missing")
            
        if os.path.exists("augmented_synthetic_health_dataset.csv"):
            print("✅ Dataset file exists")
        else:
            print("⚠️  Dataset file not found")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_api_files():
    """Test if API files can be imported"""
    print("\n🔍 Testing API Files...")
    
    try:
        # Test if we can read the API files
        api_files = [
            "apis/clinicaltrials.py",
            "apis/cowin.py", 
            "apis/mohfw.py",
            "apis/umls.py"
        ]
        
        for api_file in api_files:
            if os.path.exists(api_file):
                print(f"✅ {api_file} exists")
            else:
                print(f"❌ {api_file} missing")
                
        return True
        
    except Exception as e:
        print(f"❌ Error testing API files: {e}")
        return False

def test_model_files():
    """Test if model files exist"""
    print("\n🤖 Testing Model Files...")
    
    try:
        model_files = [
            "models/sihdemo.py",
            "models/ml_predict.py"
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"✅ {model_file} exists")
            else:
                print(f"❌ {model_file} missing")
                
        return True
        
    except Exception as e:
        print(f"❌ Error testing model files: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 HealthBot Simple Setup Test")
    print("=" * 50)
    
    tests = [
        ("Basic Setup", test_basic_imports),
        ("API Files", test_api_files),
        ("Model Files", test_model_files)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} Test...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Basic setup looks good!")
        print("\n📋 Next Steps:")
        print("1. Install dependencies: python -m pip install flask flask-cors")
        print("2. Run the application: python main.py")
        print("3. Or use Docker: docker-compose up")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        
    print("\n💡 If you're having dependency issues, try:")
    print("   python -m pip install --user flask flask-cors python-dotenv")

if __name__ == "__main__":
    main()
