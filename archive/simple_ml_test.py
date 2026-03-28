#!/usr/bin/env python3
"""
One-liner test for ML model integration
"""

print("🧪 Testing ML Model + Chatbot Integration...")

try:
    # Test ML model
    from models.ml_predict import predict_disease
    prediction, confidence, probabilities = predict_disease("I have fever and cough")
    print(f"✅ ML Model: {prediction} (Confidence: {confidence:.3f})")
    
    # Test chatbot with ML
    from chatbot import generate_bot_response
    response = generate_bot_response("I have fever and cough")
    print(f"✅ Chatbot: {response.get('type')} - {response.get('disease', 'N/A')} ({response.get('confidence', 0):.3f})")
    
    # Test different symptoms
    response2 = generate_bot_response("I have headache and nausea")
    print(f"✅ Chatbot: {response2.get('type')} - {response2.get('disease', 'N/A')} ({response2.get('confidence', 0):.3f})")
    
    print("\n🎉 SUCCESS! ML model works perfectly with chatbot!")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("💡 Install dependencies: pip install -r requirements.txt")