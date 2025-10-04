#!/usr/bin/env python3
"""
Quick prediction using the trained TF-IDF + Random Forest model
"""
import os
import joblib
import re

# Global model variables
_model = None
_vectorizer = None
_label_encoder = None

def load_quick_model():
    """Load the quick trained model"""
    global _model, _vectorizer, _label_encoder
    
    try:
        if os.path.exists('quick_trained_model.joblib'):
            bundle = joblib.load('quick_trained_model.joblib')
            _model = bundle['model']
            _vectorizer = bundle['vectorizer']
            _label_encoder = bundle['label_encoder']
            print("✅ Quick model loaded successfully!")
            return True
        else:
            print("❌ Quick model file not found")
            return False
    except Exception as e:
        print(f"❌ Error loading quick model: {e}")
        return False

def predict_new(text, numeric_feats=None):
    """
    Predict disease from symptoms using the quick trained model
    
    Args:
        text: Symptom description
        numeric_feats: Optional numeric features (not used in quick model)
    
    Returns:
        tuple: (predicted_label, confidence, probabilities)
    """
    global _model, _vectorizer, _label_encoder
    
    # Load model if not loaded
    if _model is None:
        if not load_quick_model():
            return "Model not available", 0.0, []
    
    try:
        # Clean text
        cleaned_text = clean_text(text)
        
        # Vectorize
        X_vec = _vectorizer.transform([cleaned_text])
        
        # Predict
        pred = _model.predict(X_vec)[0]
        pred_label = _label_encoder.inverse_transform([pred])[0]
        confidence = _model.predict_proba(X_vec)[0].max()
        probabilities = _model.predict_proba(X_vec)[0]
        
        return pred_label, float(confidence), probabilities
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return "Error", 0.0, []

def clean_text(text):
    """Clean text for prediction"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def simple_tokenize(text, lang="en"):
    """Simple tokenization for compatibility"""
    return clean_text(text)

def detect_lang(text):
    """Simple language detection"""
    # Very basic - just return 'en' for now
    return "en"

# Test the model
if __name__ == "__main__":
    print("🧪 Testing quick model...")
    
    if load_quick_model():
        test_cases = [
            "fever and cough with headache",
            "chest pain and shortness of breath",
            "nausea and vomiting",
            "rash and itching"
        ]
        
        for test_text in test_cases:
            label, conf, probs = predict_new(test_text)
            print(f"'{test_text}' -> {label} (confidence: {conf:.3f})")
    else:
        print("❌ Could not load model")
