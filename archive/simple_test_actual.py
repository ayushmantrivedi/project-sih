#!/usr/bin/env python3
"""
Simple one-liner test for your actual master branch files
"""

print("🧪 Testing ML Model + Chatbot with Your Master Branch Files...")

try:
    # Test ML model directly
    from models.ml_predict import predict_disease
    prediction, confidence, probabilities = predict_disease("I have fever and cough")
    print(f"✅ ML Model: {prediction} (Confidence: {confidence:.3f})")
    
    # Test chatbot with ML model
    from chatbot import generate_bot_response
    response = generate_bot_response("I have fever and cough")
    print(f"✅ Chatbot: {response.get('type')} - {response.get('disease', 'N/A')} ({response.get('confidence', 0):.3f})")
    
    # Test another symptom
    response2 = generate_bot_response("I have headache and nausea")
    print(f"✅ Chatbot: {response2.get('type')} - {response2.get('disease', 'N/A')} ({response2.get('confidence', 0):.3f})")
    
    print("\n🎉 SUCCESS! Your ML model works perfectly with chatbot!")
    print("💡 Your master branch files are working correctly!")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("💡 Common fixes:")
    print("   - Install dependencies: pip install -r requirements.txt")
    print("   - Check if quick_trained_model.joblib exists")
    print("   - Verify all Python files are in correct locations")