#!/usr/bin/env python3
"""
Enhanced HealthBot AI Chatbot Application with ML Model Integration
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

# Import utilities
from utils.model_loader import get_model_loader, load_model, predict_disease
from utils import get_logger

logger = get_logger("enhanced_app")

# Initialize model loader
model_loader = get_model_loader()

def generate_enhanced_response(user_message):
    """Generate enhanced responses with ML model integration"""
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
            "message": "Hello! I'm your AI-powered health assistant. I can help you with symptom analysis, medical information, and health data. How can I help you today?"
        }
    else:
        # Use ML model for symptom analysis
        try:
            if model_loader.is_loaded:
                # Make ML prediction
                label, confidence, probabilities = predict_disease(user_message)
                
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
                    "probabilities": probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities,
                    "message": f"Based on your symptoms, I suggest: {label} (Confidence: {confidence:.1%})",
                    "disclaimer": disclaimer,
                    "model_info": {
                        "model_type": "Bio-ClinicalBERT",
                        "version": "1.0.0"
                    }
                }
            else:
                # Fallback to simple responses if model not loaded
                return generate_simple_fallback(user_message)
                
        except Exception as e:
            logger.error(f"Error in ML prediction: {str(e)}")
            return {
                "type": "error",
                "message": "I'm having trouble analyzing your symptoms right now. Please try again or consult a healthcare professional.",
                "error": str(e)
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
            "model_info": {"model_type": "Simple Rule-Based", "version": "fallback"}
        }
    elif "fever" in user_message_lower:
        return {
            "type": "diagnosis", 
            "disease": "Fever",
            "confidence": 0.60,
            "confidence_level": "medium",
            "message": "Fever can have many causes. Please monitor your temperature and consult a doctor if it persists.",
            "model_info": {"model_type": "Simple Rule-Based", "version": "fallback"}
        }
    elif "headache" in user_message_lower:
        return {
            "type": "diagnosis",
            "disease": "Headache", 
            "confidence": 0.65,
            "confidence_level": "medium",
            "message": "Headaches can be caused by stress, dehydration, or other factors. Rest and stay hydrated.",
            "model_info": {"model_type": "Simple Rule-Based", "version": "fallback"}
        }
    else:
        return {
            "type": "general",
            "message": "I'm here to help with health-related questions. Please describe your symptoms in detail for better analysis."
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_info = model_loader.get_model_info()
    
    return jsonify({
        "status": "healthy",
        "version": "1.0.0-enhanced",
        "timestamp": "2024-01-01T00:00:00Z",
        "mode": "enhanced (with ML model)",
        "model": model_info
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
        
        # Generate enhanced bot response
        response = generate_enhanced_response(user_message)
        
        # Log the interaction
        logger.info(f"User {user_id}: {user_message}")
        logger.info(f"Bot response: {response.get('type', 'unknown')} - {response.get('disease', 'N/A')}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "Sorry, I encountered an error. Please try again."
        }), 500

@app.route('/model/status', methods=['GET'])
def model_status():
    """Get model status and information"""
    model_info = model_loader.get_model_info()
    return jsonify(model_info)

@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Reload the ML model"""
    try:
        success = load_model()
        if success:
            return jsonify({
                "status": "success",
                "message": "Model reloaded successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to reload model"
            }), 500
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error reloading model: {str(e)}"
        }), 500

@app.route('/webhook/whatsapp', methods=['POST'])
def whatsapp_webhook():
    """WhatsApp webhook endpoint with ML integration"""
    try:
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        
        if not incoming_msg:
            return "No message received", 400
        
        # Generate enhanced response
        response_data = generate_enhanced_response(incoming_msg)
        
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
        logger.error(f"Error in WhatsApp webhook: {str(e)}")
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
        
        # Generate enhanced response
        response_data = generate_enhanced_response(incoming_msg)
        
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
        logger.error(f"Error in SMS webhook: {str(e)}")
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
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred."
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced HealthBot AI Chatbot with ML Integration...")
    print("=" * 70)
    
    # Load the ML model
    print("🔬 Loading ML model...")
    if load_model():
        print("✅ ML model loaded successfully!")
    else:
        print("⚠️  ML model loading failed, using fallback mode")
    
    print("🌐 Web API: http://localhost:5001")
    print("❤️  Health Check: http://localhost:5001/health")
    print("💬 Chat API: POST http://localhost:5001/chat")
    print("🤖 Model Status: http://localhost:5001/model/status")
    print("🔄 Model Reload: POST http://localhost:5001/model/reload")
    print("📱 WhatsApp Webhook: POST http://localhost:5001/webhook/whatsapp")
    print("📧 SMS Webhook: POST http://localhost:5001/webhook/sms")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
