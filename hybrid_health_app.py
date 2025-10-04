#!/usr/bin/env python3
"""
Hybrid Health Application: APIs + Smart Medical Knowledge + ML
Combines all your health APIs with intelligent symptom analysis
"""
import os
import sys
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import hybrid system
from hybrid_health_system import HybridHealthSystem

# Import enhanced APIs
try:
    from apis.enhanced_clinicaltrials import EnhancedClinicalTrialsAPI
    from apis.enhanced_cowin import EnhancedCoWINAPI
    from apis.enhanced_mohfw import EnhancedMoHFWAPI
    from apis.enhanced_umls import EnhancedUMLSAPI
    print("✅ Enhanced APIs imported successfully")
except ImportError as e:
    print(f"⚠️ Enhanced APIs not available: {e}")
    EnhancedClinicalTrialsAPI = None
    EnhancedCoWINAPI = None
    EnhancedMoHFWAPI = None
    EnhancedUMLSAPI = None

# Import ML utilities
try:
    from models.quick_predict import predict_new, simple_tokenize, detect_lang, _label_encoder
    print("✅ ML utilities imported successfully")
    ML_AVAILABLE = True
except Exception as e:
    print(f"⚠️ ML utilities not available: {e}")
    ML_AVAILABLE = False
    _label_encoder = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hybrid_health_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('hybrid_health_app')

app = Flask(__name__)
CORS(app)

# Initialize components
hybrid_system = HybridHealthSystem()

# Initialize APIs
clinical_trials_api = EnhancedClinicalTrialsAPI() if EnhancedClinicalTrialsAPI else None
cowin_api = EnhancedCoWINAPI() if EnhancedCoWINAPI else None
mohfw_api = EnhancedMoHFWAPI() if EnhancedMoHFWAPI else None
umls_api = EnhancedUMLSAPI() if EnhancedUMLSAPI else None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Hybrid Health Application",
        "components": {
            "hybrid_system": "available",
            "ml_model": "available" if ML_AVAILABLE else "fallback",
            "clinical_trials_api": "available" if clinical_trials_api else "unavailable",
            "cowin_api": "available" if cowin_api else "unavailable",
            "mohfw_api": "available" if mohfw_api else "unavailable",
            "umls_api": "available" if umls_api else "unavailable"
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint with hybrid analysis"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        user_id = data.get('user_id', 'anonymous')
        
        if not user_message:
            return jsonify({
                "error": "Message is required",
                "response": "Please provide a message to analyze."
            }), 400
        
        logger.info(f"User {user_id}: {user_message}")
        
        # Use hybrid system for analysis
        analysis_result = hybrid_system.analyze_symptoms(user_message)
        
        # Generate comprehensive response
        response = generate_hybrid_response(analysis_result, user_message, user_id)
        
        logger.info(f"Bot response: {response.get('type', 'unknown')} - {response.get('primary_condition', 'N/A')}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "response": "I encountered an error processing your request. Please try again."
        }), 500

def generate_hybrid_response(analysis_result: dict, user_message: str, user_id: str) -> dict:
    """Generate comprehensive response using hybrid analysis"""
    
    response_type = analysis_result.get('type', 'general')
    
    if response_type == 'hybrid_analysis':
        # Use hybrid system result
        primary_condition = analysis_result.get('primary_condition', 'Unknown')
        urgency_level = analysis_result.get('urgency_level', 'low')
        symptoms = analysis_result.get('symptoms_detected', [])
        api_suggestions = analysis_result.get('api_suggestions', [])
        
        # Create comprehensive message
        message = analysis_result.get('message', '')
        
        # Add API integrations if suggested
        api_responses = {}
        if api_suggestions:
            message += "\n\n🔍 **Additional Information Available:**\n"
            for suggestion in api_suggestions[:3]:  # Limit to 3 suggestions
                api_name = suggestion['api']
                query = suggestion['query']
                description = suggestion['description']
                
                # Call relevant API
                api_result = call_relevant_api(api_name, query)
                if api_result:
                    api_responses[api_name] = api_result
                    message += f"• {description}\n"
        
        return {
            "type": "hybrid_diagnosis",
            "primary_condition": primary_condition,
            "urgency_level": urgency_level,
            "symptoms_detected": symptoms,
            "message": message,
            "recommendations": analysis_result.get('recommendations', []),
            "possible_conditions": analysis_result.get('possible_conditions', []),
            "api_responses": api_responses,
            "system_info": analysis_result.get('system_info', {}),
            "disclaimer": analysis_result.get('disclaimer', '')
        }
    
    else:
        # General response
        return {
            "type": "general",
            "message": analysis_result.get('message', ''),
            "suggestions": analysis_result.get('suggestions', [])
        }

def call_relevant_api(api_name: str, query: str) -> dict:
    """Call relevant API based on suggestion"""
    try:
        if api_name == "Clinical Trials" and clinical_trials_api:
            # Extract condition from query
            condition = query.replace("clinical trials for ", "").replace("clinical trials ", "")
            return clinical_trials_api.search_trials(condition)
        
        elif api_name == "Vaccination" and cowin_api:
            return cowin_api.get_vaccination_stats()
        
        elif api_name == "COVID-19" and mohfw_api:
            return mohfw_api.get_covid_stats()
        
        elif api_name == "Medical Dictionary" and umls_api:
            # Extract term from query
            term = query.replace("define ", "").replace("medical definition of ", "")
            return umls_api.get_medical_definition(term)
        
        return None
        
    except Exception as e:
        logger.warning(f"API call failed for {api_name}: {e}")
        return None

@app.route('/api/status', methods=['GET'])
def api_status():
    """Get status of all APIs"""
    status = {
        "hybrid_system": "available",
        "ml_model": "available" if ML_AVAILABLE else "fallback",
        "apis": {}
    }
    
    if clinical_trials_api:
        status["apis"]["clinical_trials"] = "available"
    if cowin_api:
        status["apis"]["cowin"] = "available"
    if mohfw_api:
        status["apis"]["mohfw"] = "available"
    if umls_api:
        status["apis"]["umls"] = "available"
    
    return jsonify(status)

@app.route('/test/symptom', methods=['POST'])
def test_symptom():
    """Test endpoint for specific symptom analysis"""
    try:
        data = request.get_json()
        symptom = data.get('symptom', '').strip()
        
        if not symptom:
            return jsonify({"error": "Symptom is required"}), 400
        
        # Analyze symptom
        result = hybrid_system.analyze_symptoms(symptom)
        
        return jsonify({
            "symptom": symptom,
            "analysis": result,
            "timestamp": "2025-10-04T12:00:00Z"
        })
        
    except Exception as e:
        logger.error(f"Error in test_symptom: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/webhook/whatsapp', methods=['POST'])
def whatsapp_webhook():
    """WhatsApp webhook endpoint"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        from_number = data.get('from', 'unknown')
        
        if message:
            # Process message with hybrid system
            analysis_result = hybrid_system.analyze_symptoms(message)
            response = generate_hybrid_response(analysis_result, message, from_number)
            
            # Format for WhatsApp
            whatsapp_response = {
                "to": from_number,
                "message": response.get('message', ''),
                "type": response.get('type', 'general')
            }
            
            return jsonify(whatsapp_response)
        
        return jsonify({"error": "No message provided"}), 400
        
    except Exception as e:
        logger.error(f"Error in WhatsApp webhook: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/webhook/sms', methods=['POST'])
def sms_webhook():
    """SMS webhook endpoint"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        from_number = data.get('from', 'unknown')
        
        if message:
            # Process message with hybrid system
            analysis_result = hybrid_system.analyze_symptoms(message)
            response = generate_hybrid_response(analysis_result, message, from_number)
            
            # Format for SMS (shorter response)
            sms_message = response.get('message', '')[:160]  # SMS limit
            if len(response.get('message', '')) > 160:
                sms_message += "..."
            
            sms_response = {
                "to": from_number,
                "message": sms_message,
                "type": response.get('type', 'general')
            }
            
            return jsonify(sms_response)
        
        return jsonify({"error": "No message provided"}), 400
        
    except Exception as e:
        logger.error(f"Error in SMS webhook: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    print("🚀 Starting Hybrid Health Application...")
    print("=" * 60)
    print("🔬 Hybrid System: Smart Medical Knowledge + APIs + ML")
    print("🌐 Web API: http://localhost:5005")
    print("❤️  Health Check: http://localhost:5005/health")
    print("💬 Chat API: POST http://localhost:5005/chat")
    print("🧪 Test Symptoms: POST http://localhost:5005/test/symptom")
    print("📊 API Status: http://localhost:5005/api/status")
    print("=" * 60)
    print("🔬 ClinicalTrials.gov - Search clinical trials")
    print("💉 CoWIN API - Vaccination statistics")
    print("🦠 MoHFW API - COVID-19 data")
    print("📖 UMLS API - Medical definitions")
    print("🤖 Hybrid System - Intelligent symptom analysis")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5005, debug=True)
