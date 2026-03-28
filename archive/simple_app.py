#!/usr/bin/env python3
"""
Simple HealthBot AI Chatbot Application
This version works without heavy ML dependencies for testing
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

# Simple chatbot responses without ML
def generate_simple_response(user_message):
    """Generate simple responses without ML model"""
    user_message_lower = user_message.lower()
    
    # Health-related responses
    if "fever" in user_message_lower and "cough" in user_message_lower:
        return {
            "type": "diagnosis",
            "disease": "Common Cold",
            "confidence": 0.75,
            "confidence_level": "medium",
            "message": "Based on your symptoms (fever and cough), this might be a common cold. Please rest and stay hydrated."
        }
    elif "fever" in user_message_lower:
        return {
            "type": "diagnosis", 
            "disease": "Fever",
            "confidence": 0.60,
            "confidence_level": "medium",
            "message": "Fever can have many causes. Please monitor your temperature and consult a doctor if it persists."
        }
    elif "headache" in user_message_lower:
        return {
            "type": "diagnosis",
            "disease": "Headache", 
            "confidence": 0.65,
            "confidence_level": "medium",
            "message": "Headaches can be caused by stress, dehydration, or other factors. Rest and stay hydrated."
        }
    elif "trial" in user_message_lower or "clinical" in user_message_lower:
        return {
            "type": "clinical_trials",
            "content": "Clinical trials information would be available here with full API integration."
        }
    elif "covid" in user_message_lower or "vaccine" in user_message_lower:
        return {
            "type": "covid",
            "content": "COVID-19 and vaccination information would be available here with full API integration."
        }
    elif "hello" in user_message_lower or "hi" in user_message_lower:
        return {
            "type": "greeting",
            "message": "Hello! I'm your health assistant. How can I help you today?"
        }
    else:
        return {
            "type": "general",
            "message": "I'm here to help with health-related questions. Please describe your symptoms or ask about health information."
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0-simple",
        "timestamp": "2024-01-01T00:00:00Z",
        "mode": "simple (without ML)"
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
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
        
        # Generate bot response
        response = generate_simple_response(user_message)
        
        # Log the interaction (simple)
        print(f"User {user_id}: {user_message}")
        print(f"Bot response: {response}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "Sorry, I encountered an error. Please try again."
        }), 500

@app.route('/webhook/whatsapp', methods=['POST'])
def whatsapp_webhook():
    """WhatsApp webhook endpoint (simplified)"""
    try:
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        
        if not incoming_msg:
            return "No message received", 400
        
        # Generate response
        response_data = generate_simple_response(incoming_msg)
        
        # Format response for WhatsApp
        if response_data.get('type') == 'diagnosis':
            reply = f"HealthBot: {response_data.get('message', 'Unable to process request.')}"
        else:
            reply = f"HealthBot: {response_data.get('message', 'Unable to process request.')}"
        
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
    """SMS webhook endpoint (simplified)"""
    try:
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        
        if not incoming_msg:
            return "No message received", 400
        
        # Generate response
        response_data = generate_simple_response(incoming_msg)
        
        # Format response for SMS (shorter)
        if response_data.get('type') == 'diagnosis':
            reply = f"HealthBot: {response_data.get('message', 'Unable to process.')}"
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
    print("🚀 Starting HealthBot AI Chatbot (Simple Mode)...")
    print("🌐 Web API: http://localhost:5000")
    print("❤️  Health Check: http://localhost:5000/health")
    print("💬 Chat API: POST http://localhost:5000/chat")
    print("📱 WhatsApp Webhook: POST http://localhost:5000/webhook/whatsapp")
    print("📧 SMS Webhook: POST http://localhost:5000/webhook/sms")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
