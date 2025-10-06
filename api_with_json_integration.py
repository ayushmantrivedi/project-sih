#!/usr/bin/env python3
"""
API integration example that uses the enhanced ML model with JSON data integration
"""

import os
import sys
import json
from flask import Flask, request, jsonify
from typing import Dict, Any, Optional

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from sihdemo import predict_new, load_trained_model

app = Flask(__name__)

# Global model state
model_loaded = False

def ensure_model_loaded():
    """Ensure the model is loaded before making predictions"""
    global model_loaded
    if not model_loaded:
        print("🔄 Loading trained model...")
        success = load_trained_model("sih_model_bundle.joblib")
        if success:
            model_loaded = True
            print("✅ Model loaded successfully")
        else:
            print("❌ Failed to load model")
    return model_loaded

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "message": "API is running with JSON data integration"
    })

@app.route('/predict', methods=['POST'])
def predict_disease():
    """
    Predict disease from symptoms
    
    Expected JSON payload:
    {
        "symptoms": "fever headache body ache",
        "numeric_features": [25, 1, 7]  # optional: age, gender_encoded, severity
    }
    """
    try:
        # Ensure model is loaded
        if not ensure_model_loaded():
            return jsonify({
                "error": "Model not available",
                "message": "Please train the model first"
            }), 500
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
        
        # Extract symptoms
        symptoms = data.get('symptoms', '')
        if not symptoms:
            return jsonify({
                "error": "No symptoms provided"
            }), 400
        
        # Extract optional numeric features
        numeric_features = data.get('numeric_features', None)
        
        # Make prediction
        predicted_disease, confidence, probabilities = predict_new(symptoms, numeric_features)
        
        # Prepare response
        response = {
            "symptoms": symptoms,
            "predicted_disease": predicted_disease,
            "confidence": round(confidence, 4),
            "probabilities": probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities,
            "numeric_features_used": numeric_features is not None
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict diseases for multiple symptom sets
    
    Expected JSON payload:
    {
        "symptoms_list": [
            {"symptoms": "fever headache", "numeric_features": [25, 1, 7]},
            {"symptoms": "cough chest pain", "numeric_features": [45, 0, 8]}
        ]
    }
    """
    try:
        # Ensure model is loaded
        if not ensure_model_loaded():
            return jsonify({
                "error": "Model not available",
                "message": "Please train the model first"
            }), 500
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
        
        symptoms_list = data.get('symptoms_list', [])
        if not symptoms_list:
            return jsonify({
                "error": "No symptoms list provided"
            }), 400
        
        # Process each symptom set
        results = []
        for i, item in enumerate(symptoms_list):
            symptoms = item.get('symptoms', '')
            numeric_features = item.get('numeric_features', None)
            
            if not symptoms:
                results.append({
                    "index": i,
                    "error": "No symptoms provided"
                })
                continue
            
            try:
                predicted_disease, confidence, probabilities = predict_new(symptoms, numeric_features)
                results.append({
                    "index": i,
                    "symptoms": symptoms,
                    "predicted_disease": predicted_disease,
                    "confidence": round(confidence, 4),
                    "probabilities": probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities,
                    "numeric_features_used": numeric_features is not None
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "symptoms": symptoms,
                    "error": str(e)
                })
        
        return jsonify({
            "results": results,
            "total_processed": len(results)
        })
        
    except Exception as e:
        return jsonify({
            "error": "Batch prediction failed",
            "message": str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        "model_loaded": model_loaded,
        "features": {
            "supports_json_data": True,
            "supports_numeric_features": True,
            "supports_batch_prediction": True,
            "supports_multilingual": True
        },
        "data_sources": {
            "csv_data": "C:\\Users\\ayush\\OneDrive\\Desktop\\augmented_synthetic_health_dataset.csv",
            "json_data": "/workspace/diagnosis_data.json"
        }
    })

if __name__ == '__main__':
    print("🚀 Starting API server with JSON data integration...")
    print("📊 Model will be loaded on first prediction request")
    print("🌐 API endpoints available:")
    print("   - GET  /health")
    print("   - POST /predict")
    print("   - POST /predict_batch")
    print("   - GET  /model_info")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=True)