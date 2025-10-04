#!/usr/bin/env python3
"""
HealthBot AI Chatbot Startup Script
"""
import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Error: Python 3.9 or higher is required")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import tensorflow
        import transformers
        import twilio
        print("✅ Core dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

def check_configuration():
    """Check if configuration is valid"""
    try:
        from config import validate_config
        if validate_config():
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration validation failed")
            print("Please check your .env file and configuration")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)

def check_model_files():
    """Check if model files exist"""
    from config import get_config
    config = get_config()
    
    model_files = [
        config.ml.MODEL_WEIGHTS_PATH,
        config.ml.MODEL_BUNDLE_PATH
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Warning: Some model files are missing:")
        for file in missing_files:
            print(f"   - {file}")
        print("You may need to train the model first:")
        print("   python models/sihdemo.py --csv augmented_synthetic_health_dataset.csv")
    else:
        print("✅ Model files are present")

def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "models", 
        "database",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created")

def main():
    """Main startup function"""
    print("🚀 Starting HealthBot AI Chatbot...")
    print("=" * 50)
    
    # Run checks
    check_python_version()
    check_dependencies()
    create_directories()
    check_model_files()
    check_configuration()
    
    print("=" * 50)
    print("✅ All checks passed! Starting application...")
    print("🌐 Web API: http://localhost:5000")
    print("❤️  Health Check: http://localhost:5000/health")
    print("💬 Chat API: POST http://localhost:5000/chat")
    print("📱 WhatsApp Webhook: POST http://localhost:5000/webhook/whatsapp")
    print("📧 SMS Webhook: POST http://localhost:5000/webhook/sms")
    print("=" * 50)
    
    # Start the application
    try:
        from main import app
        from config import get_config
        config = get_config()
        
        app.run(
            host=config.app.HOST,
            port=config.app.PORT,
            debug=config.app.DEBUG
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down HealthBot...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

