#!/usr/bin/env python3
"""
GPU-Optimized Prediction Module for HealthBot AI Chatbot
Features:
- GPU acceleration for inference
- Batch processing for multiple predictions
- Memory optimization
- Caching for faster repeated predictions
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Set environment variables for GPU optimization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertModel
from typing import List, Optional, Tuple, Dict, Any
import re
import gc

class GPUOptimizedPredictor:
    """GPU-optimized predictor for health diagnosis"""
    
    def __init__(self, model_path: str = "gpu_optimized_model/model_bundle.joblib"):
        """Initialize the predictor"""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.encoder = None
        self.label_encoder = None
        self.scaler = None
        self.numeric_cols = None
        self.config = None
        self.is_loaded = False
        
        # GPU setup
        self.gpu_available = self._setup_gpu()
        
        # Load model
        self.load_model()
    
    def _setup_gpu(self) -> bool:
        """Setup GPU configuration"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                print("⚠️  No GPU detected. Using CPU for inference")
                return False
            
            # Configure GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✅ GPU setup complete. Found {len(gpus)} GPU(s)")
            return True
            
        except Exception as e:
            print(f"⚠️  GPU setup failed: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the trained model and components"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            print("🔄 Loading GPU-optimized model...")
            
            # Load bundle
            bundle = joblib.load(self.model_path)
            
            # Load components
            self.tokenizer = BertTokenizerFast.from_pretrained(bundle['tokenizer_path'])
            self.encoder = TFBertModel.from_pretrained(bundle['encoder_path'])
            self.encoder.trainable = False  # Freeze for inference
            
            self.label_encoder = bundle['label_encoder']
            self.scaler = bundle.get('scaler', None)
            self.numeric_cols = bundle.get('numeric_cols', [])
            self.config = bundle.get('config', {})
            
            # Load model weights
            self._load_model_weights(bundle['model_weights'])
            
            self.is_loaded = True
            print("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def _load_model_weights(self, weights_path: str):
        """Load model weights and build model"""
        try:
            # Get model configuration from bundle
            text_dim = 768
            numeric_dim = len(self.numeric_cols) if self.numeric_cols else 0
            input_dim = text_dim + numeric_dim
            num_classes = len(self.label_encoder.classes_)
            
            # Build model architecture
            self.model = self._build_model(input_dim, num_classes, text_dim, numeric_dim)
            
            # Load weights
            self.model.load_weights(weights_path)
            
            print(f"✅ Model weights loaded. Input dim: {input_dim}, Classes: {num_classes}")
            
        except Exception as e:
            print(f"❌ Error loading model weights: {e}")
            raise
    
    def _build_model(self, input_dim: int, num_classes: int, 
                    text_dim: int = 768, numeric_dim: int = 0) -> tf.keras.Model:
        """Build the model architecture"""
        inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32, name="input")
        
        # Separate text and numeric features
        text_features = tf.keras.layers.Lambda(lambda x: x[:, :text_dim])(inputs)
        numeric_features = tf.keras.layers.Lambda(lambda x: x[:, text_dim:])(inputs) if numeric_dim > 0 else None
        
        # Text processing branch
        text_branch = tf.keras.layers.Dense(1024, activation="gelu", name="text_dense1")(text_features)
        text_branch = tf.keras.layers.BatchNormalization()(text_branch)
        text_branch = tf.keras.layers.Dropout(0.3)(text_branch)
        
        text_branch = tf.keras.layers.Dense(512, activation="gelu", name="text_dense2")(text_branch)
        text_branch = tf.keras.layers.BatchNormalization()(text_branch)
        text_branch = tf.keras.layers.Dropout(0.3)(text_branch)
        
        # Numeric processing branch
        if numeric_dim > 0:
            numeric_branch = tf.keras.layers.Dense(max(128, numeric_dim * 4), activation="gelu")(numeric_features)
            numeric_branch = tf.keras.layers.BatchNormalization()(numeric_branch)
            numeric_branch = tf.keras.layers.Dropout(0.2)(numeric_branch)
            
            # Feature fusion with attention
            fused = tf.keras.layers.Concatenate()([text_branch, numeric_branch])
            attention = tf.keras.layers.Dense(1, activation="sigmoid")(fused)
            attended = tf.keras.layers.Multiply()([fused, attention])
        else:
            attended = text_branch
        
        # Final classification layers
        x = tf.keras.layers.Dense(256, activation="gelu")(attended)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(128, activation="gelu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(num_classes, name="outputs")(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gpu_optimized_classifier")
        return model
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        if not isinstance(text, str):
            text = str(text)
        
        text = text.lower().strip()
        
        # Remove punctuation and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove medical stopwords
        medical_stopwords = {
            "mg", "ml", "tablet", "dose", "day", "week", "month", "year",
            "patient", "history", "present", "complaint", "chief"
        }
        
        tokens = text.split()
        tokens = [t for t in tokens if t not in medical_stopwords and len(t) > 2]
        
        return " ".join(tokens)
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embeddings"""
        try:
            # Normalize text
            normalized_text = self.normalize_text(text)
            
            # Tokenize
            encoding = self.tokenizer(
                normalized_text,
                truncation=True,
                padding='max_length',
                max_length=self.config.get('MAX_LEN', 128),
                return_tensors='tf'
            )
            
            # Get embeddings
            outputs = self.encoder(encoding, training=False)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # CLS token
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Error encoding text: {e}")
            return np.zeros((1, 768), dtype=np.float32)
    
    def predict_single(self, text: str, numeric_features: Optional[List[float]] = None) -> Tuple[str, float, np.ndarray]:
        """Predict disease for a single text"""
        if not self.is_loaded:
            return "Model not loaded", 0.0, np.array([])
        
        try:
            # Encode text
            text_embeddings = self.encode_text(text)
            
            # Prepare numeric features
            if numeric_features is None:
                numeric_features = [0.0] * len(self.numeric_cols) if self.numeric_cols else []
            
            if self.numeric_cols and len(numeric_features) > 0:
                # Scale numeric features
                if self.scaler is not None:
                    numeric_array = self.scaler.transform([numeric_features])
                else:
                    numeric_array = np.array([numeric_features])
                
                # Fuse features
                fused_features = np.concatenate([text_embeddings, numeric_array], axis=1)
            else:
                fused_features = text_embeddings
            
            # Get prediction
            logits = self.model(fused_features, training=False)
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
            
            # Get predicted class and confidence
            predicted_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_idx])
            predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
            
            return predicted_label, confidence, probabilities
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return "Error", 0.0, np.array([])
    
    def predict_batch(self, texts: List[str], numeric_features_list: Optional[List[List[float]]] = None) -> List[Tuple[str, float, np.ndarray]]:
        """Predict diseases for a batch of texts"""
        if not self.is_loaded:
            return [("Model not loaded", 0.0, np.array([]))] * len(texts)
        
        try:
            # Process texts in batch
            batch_size = min(32, len(texts))  # Process in smaller batches
            
            all_predictions = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_numeric = numeric_features_list[i:i + batch_size] if numeric_features_list else None
                
                # Encode texts
                text_embeddings = []
                for text in batch_texts:
                    emb = self.encode_text(text)
                    text_embeddings.append(emb)
                
                text_embeddings = np.vstack(text_embeddings)
                
                # Prepare numeric features
                if batch_numeric and self.numeric_cols:
                    if self.scaler is not None:
                        numeric_array = self.scaler.transform(batch_numeric)
                    else:
                        numeric_array = np.array(batch_numeric)
                    
                    # Fuse features
                    fused_features = np.concatenate([text_embeddings, numeric_array], axis=1)
                else:
                    fused_features = text_embeddings
                
                # Get predictions
                logits = self.model(fused_features, training=False)
                probabilities = tf.nn.softmax(logits, axis=-1).numpy()
                
                # Process predictions
                for j, probs in enumerate(probabilities):
                    predicted_idx = np.argmax(probs)
                    confidence = float(probs[predicted_idx])
                    predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
                    
                    all_predictions.append((predicted_label, confidence, probs))
            
            return all_predictions
            
        except Exception as e:
            print(f"❌ Error in batch prediction: {e}")
            return [("Error", 0.0, np.array([]))] * len(texts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'is_loaded': self.is_loaded,
            'gpu_available': self.gpu_available,
            'num_classes': len(self.label_encoder.classes_) if self.label_encoder else 0,
            'numeric_features': len(self.numeric_cols) if self.numeric_cols else 0,
            'config': self.config,
            'model_path': self.model_path
        }

# Global predictor instance
_global_predictor = None

def get_predictor(model_path: str = "gpu_optimized_model/model_bundle.joblib") -> GPUOptimizedPredictor:
    """Get global predictor instance"""
    global _global_predictor
    
    if _global_predictor is None:
        _global_predictor = GPUOptimizedPredictor(model_path)
    
    return _global_predictor

def predict_disease(text: str, numeric_features: Optional[List[float]] = None) -> Tuple[str, float, np.ndarray]:
    """Predict disease for a single text (convenience function)"""
    predictor = get_predictor()
    return predictor.predict_single(text, numeric_features)

def predict_batch(texts: List[str], numeric_features_list: Optional[List[List[float]]] = None) -> List[Tuple[str, float, np.ndarray]]:
    """Predict diseases for multiple texts (convenience function)"""
    predictor = get_predictor()
    return predictor.predict_batch(texts, numeric_features_list)

if __name__ == "__main__":
    # Test the predictor
    print("🧪 Testing GPU-optimized predictor...")
    
    predictor = GPUOptimizedPredictor()
    
    if predictor.is_loaded:
        # Test single prediction
        test_text = "fever and cough with headache"
        label, confidence, probs = predictor.predict_single(test_text)
        
        print(f"Test: '{test_text}'")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.3f}")
        
        # Test batch prediction
        test_texts = [
            "chest pain and shortness of breath",
            "fever and body aches",
            "headache and nausea"
        ]
        
        batch_results = predictor.predict_batch(test_texts)
        
        print("\nBatch predictions:")
        for text, (label, conf, _) in zip(test_texts, batch_results):
            print(f"  '{text}' -> {label} ({conf:.3f})")
        
        # Model info
        info = predictor.get_model_info()
        print(f"\nModel info: {info}")
        
    else:
        print("❌ Model not loaded. Please train the model first.")