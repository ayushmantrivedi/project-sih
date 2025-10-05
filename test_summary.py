#!/usr/bin/env python3
"""
Comprehensive test summary for HealthBot AI Chatbot
"""

import os
import sys
import subprocess
import traceback

def run_test(test_file):
    """Run a test file and return results"""
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Test timed out"
    except Exception as e:
        return False, "", str(e)

def main():
    """Run all tests and provide comprehensive summary"""
    print("🚀 HealthBot AI Chatbot - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", "test_basic_functionality.py"),
        ("ML Model", "test_model_non_interactive.py"),
        ("API Integrations", "test_apis_non_interactive.py"),
        ("Chatbot Core", "test_chatbot_core.py"),
        ("Web Endpoints", "test_web_endpoints.py")
    ]
    
    results = []
    total_tests = len(tests)
    passed_tests = 0
    
    print(f"Running {total_tests} test suites...\n")
    
    for test_name, test_file in tests:
        print(f"🧪 Running {test_name} tests...")
        
        if not os.path.exists(test_file):
            print(f"❌ {test_file} not found")
            results.append((test_name, False, "Test file not found", ""))
            continue
        
        success, stdout, stderr = run_test(test_file)
        
        if success:
            print(f"✅ {test_name} tests passed")
            passed_tests += 1
        else:
            print(f"❌ {test_name} tests failed")
            if stderr:
                print(f"   Error: {stderr}")
        
        results.append((test_name, success, stdout, stderr))
        print()
    
    # Print detailed results
    print("=" * 60)
    print("📊 DETAILED TEST RESULTS")
    print("=" * 60)
    
    for test_name, success, stdout, stderr in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n{test_name}: {status}")
        
        if not success and stderr:
            print(f"Error: {stderr}")
        
        # Extract key information from stdout
        if "Test Results:" in stdout:
            lines = stdout.split('\n')
            for line in lines:
                if "Test Results:" in line:
                    print(f"   {line.strip()}")
                    break
    
    # Overall summary
    print("\n" + "=" * 60)
    print("🎯 OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total Test Suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📋 Your HealthBot AI Chatbot is ready for deployment!")
        print("\n✅ Project Structure: Complete")
        print("✅ ML Model: Ready")
        print("✅ API Integrations: Ready")
        print("✅ Chatbot Logic: Ready")
        print("✅ Web Endpoints: Ready")
        print("✅ WhatsApp/SMS: Ready")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up Twilio credentials in .env file")
        print("3. Test locally: python main.py")
        print("4. Deploy to production using the deployment guide")
        
        print("\n💡 QUICK START COMMANDS:")
        print("   # Install dependencies")
        print("   pip install -r requirements.txt")
        print("   ")
        print("   # Test the chatbot")
        print("   python -c \"from chatbot import generate_bot_response; print(generate_bot_response('I have fever and cough'))\"")
        print("   ")
        print("   # Run the web server")
        print("   python main.py")
        print("   ")
        print("   # Test health endpoint")
        print("   curl http://localhost:5000/health")
        
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test suite(s) failed.")
        print("Please review the failed tests above and fix any issues.")
    
    print("\n" + "=" * 60)
    print("📚 For detailed deployment instructions, see DEPLOYMENT_GUIDE.md")
    print("📚 For Docker setup, see DOCKER_SETUP.md")
    print("=" * 60)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)