#!/usr/bin/env python3
"""
HealthBot AI Chatbot with ML Model Integration
This version integrates your Bio-ClinicalBERT model
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

# Import your ML model
try:
    from models.ml_predict import predict_disease as ml_predict_disease
    ML_AVAILABLE = True
    print("✅ ML model imported successfully")
except Exception as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML model not available: {e}")

def generate_ml_response(user_message):
    """Generate responses with ML model integration"""
    user_message_lower = user_message.lower()
    
    # Check for specific API queries first
    if "trial" in user_message_lower or "clinical trial" in user_message_lower:
        return {
            "type": "clinical_trials",
            "content": "Clinical trials information would be available here with full API integration.",
            "message": "I can help you find clinical trials. Please provide more specific details about what you're looking for."
        }
    elif "cowin" in user_message_lower or "vaccine" in user_message_lower or "vaccination" in user_message_lower:
        return {
            "type": "cowin",
            "content": "Vaccination information would be available here with full API integration.",
            "message": "I can provide vaccination statistics and information. What specific details would you like?"
        }
    elif "covid" in user_message_lower or "mohfw" in user_message_lower or "coronavirus" in user_message_lower:
        return {
            "type": "covid",
            "content": "COVID-19 information would be available here with full API integration.",
            "message": "I can provide COVID-19 statistics and information. What would you like to know?"
        }
    elif "define" in user_message_lower or "what is" in user_message_lower:
        return {
            "type": "umls",
            "content": "Medical definition would be available here with UMLS integration.",
            "message": "I can provide medical definitions. What term would you like me to define?"
        }
    elif "hello" in user_message_lower or "hi" in user_message_lower or "hey" in user_message_lower:
        return {
            "type": "greeting",
            "message": "Hello! I'm your AI-powered health assistant with Bio-ClinicalBERT model. I can help you with symptom analysis, medical information, and health data. How can I help you today?"
        }
    else:
        # Use ML model for symptom analysis
        try:
            if ML_AVAILABLE:
                # Make ML prediction using your model
                label, confidence, probabilities = ml_predict_disease(user_message)
                
                # Determine confidence level
                if confidence >= 0.8:
                    confidence_level = "high"
                    disclaimer = ""
                elif confidence >= 0.6:
                    confidence_level = "medium"
                    disclaimer = "This is a medium-confidence prediction. Please consult a healthcare professional for proper diagnosis."
                else:
                    confidence_level = "low"
                    disclaimer = "This is a low-confidence prediction. Please consult a healthcare professional for proper diagnosis."
                
                return {
                    "type": "diagnosis",
                    "disease": label,
                    "confidence": float(confidence),
                    "confidence_level": confidence_level,
                    "probabilities": probabilities.tolist() if hasattr(probabilities, 'tolist') else list(probabilities),
                    "message": f"Based on your symptoms, I suggest: {label} (Confidence: {confidence:.1%})",
                    "disclaimer": disclaimer,
                    "model_info": {
                        "model_type": "Bio-ClinicalBERT",
                        "version": "1.0.0",
                        "available": True
                    }
                }
            else:
                # Fallback to simple responses if model not available
                return generate_simple_fallback(user_message)
                
        except Exception as e:
            print(f"Error in ML prediction: {str(e)}")
            return {
                "type": "error",
                "message": "I'm having trouble analyzing your symptoms right now. Please try again or consult a healthcare professional.",
                "error": str(e),
                "model_info": {
                    "model_type": "Bio-ClinicalBERT",
                    "version": "1.0.0",
                    "available": False,
                    "error": str(e)
                }
            }

def generate_simple_fallback(user_message):
    """Simple fallback responses when ML model is not available"""
    user_message_lower = user_message.lower()
    
    if "fever" in user_message_lower and "cough" in user_message_lower:
        return {
            "type": "diagnosis",
            "disease": "Common Cold",
            "confidence": 0.75,
            "confidence_level": "medium",
            "message": "Based on your symptoms (fever and cough), this might be a common cold. Please rest and stay hydrated.",
            "model_info": {"model_type": "Simple Rule-Based", "version": "fallback", "available": False}
        }
    elif "fever" in user_message_lower:
        return {
            "type": "diagnosis", 
            "disease": "Fever",
            "confidence": 0.60,
            "confidence_level": "medium",
            "message": "Fever can have many causes. Please monitor your temperature and consult a doctor if it persists.",
            "model_info": {"model_type": "Simple Rule-Based", "version": "fallback", "available": False}
        }
    elif "headache" in user_message_lower:
        return {
            "type": "diagnosis",
            "disease": "Headache", 
            "confidence": 0.65,
            "confidence_level": "medium",
            "message": "Headaches can be caused by stress, dehydration, or other factors. Rest and stay hydrated.",
            "model_info": {"model_type": "Simple Rule-Based", "version": "fallback", "available": False}
        }
    else:
        return {
            "type": "general",
            "message": "I'm here to help with health-related questions. Please describe your symptoms in detail for better analysis."
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0-ml-integrated",
        "timestamp": "2024-01-01T00:00:00Z",
        "mode": "ML-integrated (Bio-ClinicalBERT)",
        "ml_available": ML_AVAILABLE,
        "model_info": {
            "model_type": "Bio-ClinicalBERT",
            "version": "1.0.0",
            "available": ML_AVAILABLE
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint with ML integration"""
    try:
        data = request.get_json(force=True)
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Missing 'message' field in request body"
            }), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                "error": "Empty message"
            }), 400
        
        # Get user ID for logging (if provided)
        user_id = data.get('user_id', 'anonymous')
        
        # Generate ML-powered bot response
        response = generate_ml_response(user_message)
        
        # Log the interaction
        print(f"User {user_id}: {user_message}")
        print(f"Bot response: {response.get('type', 'unknown')} - {response.get('disease', 'N/A')}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "Sorry, I encountered an error. Please try again."
        }), 500

@app.route('/model/status', methods=['GET'])
def model_status():
    """Get model status and information"""
    return jsonify({
        "model_type": "Bio-ClinicalBERT",
        "version": "1.0.0",
        "available": ML_AVAILABLE,
        "status": "loaded" if ML_AVAILABLE else "not_available"
    })

@app.route('/webhook/whatsapp', methods=['POST'])
def whatsapp_webhook():
    """WhatsApp webhook endpoint with ML integration"""
    try:
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        
        if not incoming_msg:
            return "No message received", 400
        
        # Generate ML-powered response
        response_data = generate_ml_response(incoming_msg)
        
        # Format response for WhatsApp
        if response_data.get('type') == 'diagnosis':
            reply = f"🏥 HealthBot Analysis:\n\n"
            reply += f"🔍 Condition: {response_data.get('disease', 'Unknown')}\n"
            reply += f"📊 Confidence: {response_data.get('confidence', 0):.1%}\n"
            reply += f"💡 Message: {response_data.get('message', '')}\n"
            if response_data.get('disclaimer'):
                reply += f"\n⚠️ {response_data.get('disclaimer')}"
        else:
            reply = f"🏥 HealthBot: {response_data.get('message', 'Unable to process request.')}"
        
        # Simple response format for WhatsApp
        return f"""
        <Response>
            <Message>{reply}</Message>
        </Response>
        """
        
    except Exception as e:
        print(f"Error in WhatsApp webhook: {str(e)}")
        return """
        <Response>
            <Message>Sorry, I encountered an error. Please try again.</Message>
        </Response>
        """

@app.route('/webhook/sms', methods=['POST'])
def sms_webhook():
    """SMS webhook endpoint with ML integration"""
    try:
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        
        if not incoming_msg:
            return "No message received", 400
        
        # Generate ML-powered response
        response_data = generate_ml_response(incoming_msg)
        
        # Format response for SMS (shorter)
        if response_data.get('type') == 'diagnosis':
            reply = f"HealthBot: {response_data.get('disease', 'Unknown')} ({response_data.get('confidence', 0):.1%})"
        else:
            reply = f"HealthBot: {response_data.get('message', 'Unable to process.')}"
        
        # Truncate if too long for SMS
        if len(reply) > 160:
            reply = reply[:157] + "..."
        
        # Simple response format for SMS
        return f"""
        <Response>
            <Message>{reply}</Message>
        </Response>
        """
        
    except Exception as e:
        print(f"Error in SMS webhook: {str(e)}")
        return """
        <Response>
            <Message>Error occurred. Please try again.</Message>
        </Response>
        """

@app.errorhandler(404)
def not_found_handler(e):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist."
    }), 404

@app.errorhandler(500)
def internal_error_handler(e):
    """Handle internal server errors"""
    print(f"Internal server error: {str(e)}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred."
    }), 500

if __name__ == '__main__':
    print("🚀 Starting HealthBot AI Chatbot with Bio-ClinicalBERT ML Integration...")
    print("=" * 70)
    
    if ML_AVAILABLE:
        print("✅ Bio-ClinicalBERT model loaded successfully!")
    else:
        print("⚠️  Bio-ClinicalBERT model not available, using fallback mode")
    
    print("🌐 Web API: http://localhost:5002")
    print("❤️  Health Check: http://localhost:5002/health")
    print("💬 Chat API: POST http://localhost:5002/chat")
    print("🤖 Model Status: http://localhost:5002/model/status")
    print("📱 WhatsApp Webhook: POST http://localhost:5002/webhook/whatsapp")
    print("📧 SMS Webhook: POST http://localhost:5002/webhook/sms")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5002, debug=True)
