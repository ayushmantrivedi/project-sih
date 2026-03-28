#!/usr/bin/env python3
"""
Diagnostic test to identify the "No such file or directory" error
"""

import os
import sys

def diagnose_error():
    """Diagnose the file/directory error"""
    print("🔍 Diagnosing 'No such file or directory' error...")
    print("=" * 50)
    
    # Check current directory
    print(f"\n1. Current directory: {os.getcwd()}")
    
    # Check Python path
    print(f"\n2. Python path:")
    for i, path in enumerate(sys.path):
        print(f"   {i}: {path}")
    
    # Check required files
    print(f"\n3. Checking required files:")
    required_files = [
        'models/ml_predict.py',
        'models/sihdemo.py',
        'models/__init__.py',
        'chatbot.py',
        'quick_trained_model.joblib'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file} exists")
        else:
            print(f"   ❌ {file} missing")
    
    # Check if we can import basic modules
    print(f"\n4. Testing basic imports:")
    try:
        import os
        print("   ✅ os module works")
    except Exception as e:
        print(f"   ❌ os module failed: {e}")
    
    try:
        import sys
        print("   ✅ sys module works")
    except Exception as e:
        print(f"   ❌ sys module failed: {e}")
    
    # Check if we can import from models
    print(f"\n5. Testing models import:")
    try:
        import models
        print("   ✅ models package imported")
    except Exception as e:
        print(f"   ❌ models package failed: {e}")
    
    try:
        from models import ml_predict
        print("   ✅ models.ml_predict imported")
    except Exception as e:
        print(f"   ❌ models.ml_predict failed: {e}")
    
    try:
        from models import sihdemo
        print("   ✅ models.sihdemo imported")
    except Exception as e:
        print(f"   ❌ models.sihdemo failed: {e}")
    
    # Check if we can import chatbot
    print(f"\n6. Testing chatbot import:")
    try:
        import chatbot
        print("   ✅ chatbot module imported")
    except Exception as e:
        print(f"   ❌ chatbot module failed: {e}")
    
    # Check dependencies
    print(f"\n7. Testing dependencies:")
    dependencies = ['numpy', 'pandas', 'sklearn', 'joblib', 'tensorflow', 'transformers']
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep} available")
        except ImportError:
            print(f"   ❌ {dep} not available")
    
    # Try to read the model file
    print(f"\n8. Testing model file access:")
    try:
        with open('quick_trained_model.joblib', 'rb') as f:
            header = f.read(10)
            print(f"   ✅ Model file readable (size: {os.path.getsize('quick_trained_model.joblib')} bytes)")
    except Exception as e:
        print(f"   ❌ Model file access failed: {e}")
    
    print(f"\n" + "=" * 50)
    print("🔍 Diagnosis complete!")
    print("Check the results above to identify the issue.")

if __name__ == "__main__":
    diagnose_error()