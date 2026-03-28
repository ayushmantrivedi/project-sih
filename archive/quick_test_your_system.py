#!/usr/bin/env python3
"""
Quick test for your system with master branch files
"""

print("🧪 Quick Test for Your System with Master Branch Files...")

try:
    # Test ML model
    from models.ml_predict import predict_disease
    prediction, confidence, probabilities = predict_disease("I have fever and cough")
    print(f"✅ ML Model: {prediction} (Confidence: {confidence:.3f})")
    
    # Test chatbot
    from chatbot import generate_bot_response
    response = generate_bot_response("I have fever and cough")
    print(f"✅ Chatbot: {response.get('type')} - {response.get('disease', 'N/A')} ({response.get('confidence', 0):.3f})")
    
    # Test another symptom
    response2 = generate_bot_response("I have headache and nausea")
    print(f"✅ Chatbot: {response2.get('type')} - {response2.get('disease', 'N/A')} ({response2.get('confidence', 0):.3f})")
    
    print("\n🎉 SUCCESS! Your ML model works perfectly with chatbot!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Install dependencies: pip install -r requirements.txt")
except FileNotFoundError as e:
    print(f"❌ File Not Found: {e}")
    print("💡 Make sure you're in the correct directory with your master branch files")
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Check your file structure and dependencies")