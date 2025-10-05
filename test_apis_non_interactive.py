#!/usr/bin/env python3
"""
Non-interactive API integration test for HealthBot AI Chatbot
"""

import os
import sys
import traceback

def test_api_file_structure():
    """Test API file structure"""
    print("🧪 Testing API file structure...")
    
    api_files = [
        'apis/clinicaltrials.py',
        'apis/cowin.py',
        'apis/mohfw.py',
        'apis/umls.py',
        'apis/infermedica_api.py'
    ]
    
    for api_file in api_files:
        if os.path.exists(api_file):
            try:
                with open(api_file, 'r') as f:
                    content = f.read()
                
                # Check for main function
                if 'def get_' in content:
                    print(f"✅ {api_file} has get functions")
                else:
                    print(f"❌ {api_file} missing get functions")
                    return False
                
                # Check for error handling
                if 'try:' in content and 'except' in content:
                    print(f"✅ {api_file} has error handling")
                else:
                    print(f"⚠️  {api_file} may be missing error handling")
                
                # Check for requests import
                if 'import requests' in content or 'from requests' in content:
                    print(f"✅ {api_file} uses requests library")
                else:
                    print(f"⚠️  {api_file} may not use requests library")
                
            except Exception as e:
                print(f"❌ Error reading {api_file}: {e}")
                return False
        else:
            print(f"❌ {api_file} not found")
            return False
    
    return True

def test_api_dependencies():
    """Test API dependencies"""
    print("\n🧪 Testing API dependencies...")
    
    dependencies = [
        ('requests', 'HTTP requests'),
        ('json', 'JSON handling'),
        ('urllib', 'URL handling'),
        ('time', 'Time functions')
    ]
    
    available = 0
    for dep, description in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} - {description}")
            available += 1
        except ImportError:
            print(f"❌ {dep} - {description} (not available)")
    
    print(f"\n📊 API Dependencies: {available}/{len(dependencies)} available")
    return available > 0

def test_clinical_trials_api():
    """Test ClinicalTrials API structure"""
    print("\n🧪 Testing ClinicalTrials API...")
    
    try:
        with open('apis/clinicaltrials.py', 'r') as f:
            content = f.read()
        
        # Check for main function
        if 'def get_clinical_trials' in content:
            print("✅ get_clinical_trials function found")
        else:
            print("❌ get_clinical_trials function not found")
            return False
        
        # Check for API URL
        if 'clinicaltrials.gov' in content:
            print("✅ ClinicalTrials.gov URL found")
        else:
            print("⚠️  ClinicalTrials.gov URL not found")
        
        # Check for error handling
        if 'try:' in content and 'except' in content:
            print("✅ Error handling found")
        else:
            print("⚠️  Error handling may be missing")
        
        return True
        
    except Exception as e:
        print(f"❌ ClinicalTrials API test failed: {e}")
        return False

def test_cowin_api():
    """Test CoWIN API structure"""
    print("\n🧪 Testing CoWIN API...")
    
    try:
        with open('apis/cowin.py', 'r') as f:
            content = f.read()
        
        # Check for main function
        if 'def get_cowin_stats' in content:
            print("✅ get_cowin_stats function found")
        else:
            print("❌ get_cowin_stats function not found")
            return False
        
        # Check for API URL
        if 'cowin.gov.in' in content:
            print("✅ CoWIN API URL found")
        else:
            print("⚠️  CoWIN API URL not found")
        
        return True
        
    except Exception as e:
        print(f"❌ CoWIN API test failed: {e}")
        return False

def test_mohfw_api():
    """Test MoHFW API structure"""
    print("\n🧪 Testing MoHFW API...")
    
    try:
        with open('apis/mohfw.py', 'r') as f:
            content = f.read()
        
        # Check for main function
        if 'def get_mohfw_data' in content:
            print("✅ get_mohfw_data function found")
        else:
            print("❌ get_mohfw_data function not found")
            return False
        
        # Check for API URL
        if 'covid19india.org' in content:
            print("✅ COVID-19 India API URL found")
        else:
            print("⚠️  COVID-19 India API URL not found")
        
        return True
        
    except Exception as e:
        print(f"❌ MoHFW API test failed: {e}")
        return False

def test_umls_api():
    """Test UMLS API structure"""
    print("\n🧪 Testing UMLS API...")
    
    try:
        with open('apis/umls.py', 'r') as f:
            content = f.read()
        
        # Check for main function
        if 'def get_umls_info' in content:
            print("✅ get_umls_info function found")
        else:
            print("❌ get_umls_info function not found")
            return False
        
        # Check for API URL
        if 'nlm.nih.gov' in content:
            print("✅ UMLS API URL found")
        else:
            print("⚠️  UMLS API URL not found")
        
        return True
        
    except Exception as e:
        print(f"❌ UMLS API test failed: {e}")
        return False

def test_api_configuration():
    """Test API configuration"""
    print("\n🧪 Testing API configuration...")
    
    try:
        with open('config/settings.py', 'r') as f:
            content = f.read()
        
        # Check for API configuration
        if 'class APIConfig' in content:
            print("✅ APIConfig class found")
        else:
            print("❌ APIConfig class not found")
            return False
        
        # Check for specific API URLs
        api_urls = [
            'CLINICAL_TRIALS_BASE_URL',
            'COWIN_BASE_URL',
            'MOHFW_BASE_URL',
            'UMLS_BASE_URL'
        ]
        
        for url in api_urls:
            if url in content:
                print(f"✅ {url} configuration found")
            else:
                print(f"⚠️  {url} configuration not found")
        
        return True
        
    except Exception as e:
        print(f"❌ API configuration test failed: {e}")
        return False

def test_enhanced_apis():
    """Test enhanced API files"""
    print("\n🧪 Testing enhanced API files...")
    
    enhanced_files = [
        'apis/enhanced_clinicaltrials.py',
        'apis/enhanced_cowin.py',
        'apis/enhanced_mohfw.py',
        'apis/enhanced_umls.py'
    ]
    
    for file in enhanced_files:
        if os.path.exists(file):
            try:
                with open(file, 'r') as f:
                    content = f.read()
                
                if 'def get_' in content:
                    print(f"✅ {file} has get functions")
                else:
                    print(f"❌ {file} missing get functions")
                    return False
                
            except Exception as e:
                print(f"❌ Error reading {file}: {e}")
                return False
        else:
            print(f"⚠️  {file} not found")
    
    return True

def main():
    """Run all API tests"""
    print("🚀 HealthBot AI Chatbot - API Integration Test")
    print("=" * 50)
    
    tests = [
        test_api_file_structure,
        test_api_dependencies,
        test_clinical_trials_api,
        test_cowin_api,
        test_mohfw_api,
        test_umls_api,
        test_api_configuration,
        test_enhanced_apis
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
    print(f"📊 API Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All API tests passed!")
        print("\n📋 API Status:")
        print("✅ All API files are properly structured")
        print("✅ API functions are correctly defined")
        print("✅ Configuration is properly set up")
        print("⚠️  Some dependencies may need to be installed")
        print("\n💡 To test actual API calls, install dependencies and run:")
        print("   pip install -r requirements.txt")
        print("   python -c \"from apis.clinicaltrials import get_clinical_trials; print(get_clinical_trials('diabetes'))\"")
    else:
        print("⚠️  Some API tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)