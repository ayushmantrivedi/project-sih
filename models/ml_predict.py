# Try to import GPU-optimized predictor first, fallback to original
try:
    from .gpu_optimized_predict import predict_disease as gpu_predict_disease
    GPU_OPTIMIZED_AVAILABLE = True
    print("✅ GPU-optimized predictor loaded")
except ImportError as e:
    print(f"⚠️  GPU-optimized predictor not available: {e}")
    from .sihdemo import predict_new
    GPU_OPTIMIZED_AVAILABLE = False

def predict_disease(symptom_text, numeric_features=None):
    """
    Predict disease from symptom text with optional numeric features
    
    Args:
        symptom_text: Text description of symptoms
        numeric_features: Optional list of numeric features (age, severity, etc.)
    
    Returns:
        tuple: (predicted_label, confidence, probabilities)
    """
    if GPU_OPTIMIZED_AVAILABLE:
        # Use GPU-optimized predictor
        return gpu_predict_disease(symptom_text, numeric_features)
    else:
        # Fallback to original predictor
        label, conf, probs = predict_new(symptom_text, numeric_features)
        return label, conf, probs