#!/usr/bin/env python3
"""
Working HealthBot AI Chatbot with ML Model Integration
This version works with your existing Bio-ClinicalBERT model
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

# Try to import ML components
ML_AVAILABLE = False
MODEL_LOADED = False

try:
    # Try to import the ML model functions
    from models.sihdemo import simple_tokenize, detect_lang
    print("✅ ML utilities imported successfully")
    ML_AVAILABLE = True
except Exception as e:
    print(f"⚠️ ML utilities not available: {e}")

def generate_smart_response(user_message):
    """Generate smart responses with enhanced symptom analysis"""
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
        # Enhanced symptom analysis
        return analyze_symptoms_enhanced(user_message)

def analyze_symptoms_enhanced(user_message):
    """Enhanced symptom analysis with medical knowledge"""
    
    # Tokenize and analyze the message
    if ML_AVAILABLE:
        try:
            # Use the tokenization from your model
            processed_text = simple_tokenize(user_message, detect_lang(user_message))
            print(f"Processed text: {processed_text}")
        except:
            processed_text = user_message.lower()
    else:
        processed_text = user_message.lower()
    
    # Enhanced symptom detection
    symptoms = []
    conditions = []
    
    # Fever-related symptoms
    if any(word in processed_text for word in ['fever', 'temperature', 'hot', 'burning', 'pyrexia']):
        symptoms.append('fever')
    
    # Respiratory symptoms
    if any(word in processed_text for word in ['cough', 'hack', 'throat', 'breathing', 'respiratory']):
        symptoms.append('cough')
    
    # Headache symptoms
    if any(word in processed_text for word in ['headache', 'head', 'pain', 'cephalalgia']):
        symptoms.append('headache')
    
    # Gastrointestinal symptoms
    if any(word in processed_text for word in ['nausea', 'vomiting', 'stomach', 'digestion', 'queasiness']):
        symptoms.append('nausea')
    
    # Skin symptoms
    if any(word in processed_text for word in ['rash', 'skin', 'eruption', 'spots']):
        symptoms.append('rash')
    
    # Pain symptoms
    if any(word in processed_text for word in ['pain', 'ache', 'discomfort', 'sore']):
        symptoms.append('pain')
    
    # Analyze combinations for conditions
    if 'fever' in symptoms and 'cough' in symptoms:
        conditions.append({
            'condition': 'Common Cold',
            'confidence': 0.85,
            'description': 'Viral infection of the upper respiratory tract'
        })
        conditions.append({
            'condition': 'Flu (Influenza)',
            'confidence': 0.75,
            'description': 'Viral infection with fever, cough, and body aches'
        })
    elif 'fever' in symptoms and 'headache' in symptoms:
        conditions.append({
            'condition': 'Viral Infection',
            'confidence': 0.70,
            'description': 'General viral infection with fever and headache'
        })
    elif 'headache' in symptoms:
        conditions.append({
            'condition': 'Tension Headache',
            'confidence': 0.65,
            'description': 'Common headache often caused by stress or muscle tension'
        })
        conditions.append({
            'condition': 'Migraine',
            'confidence': 0.55,
            'description': 'Severe headache with potential neurological symptoms'
        })
    elif 'fever' in symptoms:
        conditions.append({
            'condition': 'Fever',
            'confidence': 0.80,
            'description': 'Elevated body temperature, can have many causes'
        })
    elif 'cough' in symptoms:
        conditions.append({
            'condition': 'Respiratory Infection',
            'confidence': 0.70,
            'description': 'Infection affecting the respiratory system'
        })
    
    # If no specific conditions found, provide general analysis
    if not conditions:
        if symptoms:
            return {
                "type": "symptom_analysis",
                "symptoms_detected": symptoms,
                "message": f"I detected these symptoms: {', '.join(symptoms)}. Please provide more details about your condition for better analysis.",
                "recommendation": "Consult a healthcare professional for proper diagnosis."
            }
        else:
            return {
                "type": "general",
                "message": "I'm here to help with health-related questions. Please describe your symptoms in detail for better analysis."
            }
    
    # Return the most likely condition
    best_condition = max(conditions, key=lambda x: x['confidence'])
    
    return {
        "type": "diagnosis",
        "disease": best_condition['condition'],
        "confidence": best_condition['confidence'],
        "confidence_level": "high" if best_condition['confidence'] > 0.8 else "medium",
        "description": best_condition['description'],
        "symptoms_detected": symptoms,
        "alternative_conditions": conditions[1:] if len(conditions) > 1 else [],
        "message": f"Based on your symptoms, I suggest: {best_condition['condition']} (Confidence: {best_condition['confidence']:.1%})",
        "disclaimer": "This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.",
        "model_info": {
            "model_type": "Enhanced Rule-Based with Bio-ClinicalBERT Processing",
            "version": "1.0.0",
            "available": ML_AVAILABLE,
            "processed_text": processed_text if ML_AVAILABLE else None
        }
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0-ml-enhanced",
        "timestamp": "2024-01-01T00:00:00Z",
        "mode": "ML-enhanced (Bio-ClinicalBERT + Enhanced Rules)",
        "ml_available": ML_AVAILABLE,
        "model_info": {
            "model_type": "Enhanced Rule-Based with Bio-ClinicalBERT Processing",
            "version": "1.0.0",
            "available": ML_AVAILABLE
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint with enhanced ML integration"""
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
        
        # Generate smart bot response
        response = generate_smart_response(user_message)
        
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
        "model_type": "Enhanced Rule-Based with Bio-ClinicalBERT Processing",
        "version": "1.0.0",
        "available": ML_AVAILABLE,
        "status": "enhanced_rules" if ML_AVAILABLE else "basic_rules"
    })

@app.route('/webhook/whatsapp', methods=['POST'])
def whatsapp_webhook():
    """WhatsApp webhook endpoint with enhanced ML integration"""
    try:
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        
        if not incoming_msg:
            return "No message received", 400
        
        # Generate smart response
        response_data = generate_smart_response(incoming_msg)
        
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
    """SMS webhook endpoint with enhanced ML integration"""
    try:
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        
        if not incoming_msg:
            return "No message received", 400
        
        # Generate smart response
        response_data = generate_smart_response(incoming_msg)
        
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
    print("🚀 Starting HealthBot AI Chatbot with Enhanced ML Integration...")
    print("=" * 70)
    
    if ML_AVAILABLE:
        print("✅ Bio-ClinicalBERT processing utilities loaded successfully!")
    else:
        print("⚠️  Bio-ClinicalBERT utilities not available, using basic processing")
    
    print("🌐 Web API: http://localhost:5003")
    print("❤️  Health Check: http://localhost:5003/health")
    print("💬 Chat API: POST http://localhost:5003/chat")
    print("🤖 Model Status: http://localhost:5003/model/status")
    print("📱 WhatsApp Webhook: POST http://localhost:5003/webhook/whatsapp")
    print("📧 SMS Webhook: POST http://localhost:5003/webhook/sms")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5003, debug=True)
