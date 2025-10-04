"""
Model loading utilities for HealthBot AI Chatbot
"""
import os
import sys
from typing import Optional, Tuple, Any
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_logger

logger = get_logger("model_loader")

class ModelLoader:
    """Handles loading and managing the ML model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.encoder = None
        self.label_encoder = None
        self.scaler = None
        self.numeric_cols = None
        self.is_loaded = False
        
    def load_model(self, model_path: str = "models/sihdemo.py") -> bool:
        """Load the ML model and its components"""
        try:
            logger.info("Loading ML model...")
            
            # Import the model functions
            from models.sihdemo import predict_new, simple_tokenize, detect_lang
            
            # Store the prediction function
            self.predict_function = predict_new
            self.tokenize_function = simple_tokenize
            self.detect_lang_function = detect_lang
            
            self.is_loaded = True
            logger.info("ML model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ML model: {str(e)}")
            return False
    
    def predict(self, text: str, numeric_features: Optional[list] = None) -> Tuple[str, float, np.ndarray]:
        """Make a prediction using the loaded model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Use the predict_new function from sihdemo.py
            label, confidence, probabilities = self.predict_function(text, numeric_features)
            return label, confidence, probabilities
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            # Return fallback prediction
            return "Unknown", 0.0, np.array([1.0])
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "is_loaded": self.is_loaded,
            "model_type": "Bio-ClinicalBERT",
            "version": "1.0.0"
        }

# Global model loader instance
_model_loader = None

def get_model_loader() -> ModelLoader:
    """Get the global model loader instance"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader

def load_model() -> bool:
    """Load the ML model"""
    loader = get_model_loader()
    return loader.load_model()

def predict_disease(text: str, numeric_features: Optional[list] = None) -> Tuple[str, float, np.ndarray]:
    """Predict disease from symptoms"""
    loader = get_model_loader()
    return loader.predict(text, numeric_features)
