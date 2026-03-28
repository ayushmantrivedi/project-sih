#!/usr/bin/env python3
"""
HealthBot AI Chatbot with Full API Integration
This version integrates all health APIs with enhanced error handling and caching
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

# Import enhanced APIs
try:
    from apis.enhanced_clinicaltrials import search_clinical_trials
    from apis.enhanced_cowin import get_cowin_stats
    from apis.enhanced_mohfw import get_mohfw_data
    from apis.enhanced_umls import get_umls_info
    print("✅ Enhanced APIs imported successfully")
    APIS_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Enhanced APIs not available: {e}")
    APIS_AVAILABLE = False

# Import ML utilities
try:
    from models.quick_predict import predict_new, simple_tokenize, detect_lang, _label_encoder
    print("✅ ML utilities imported successfully")
    ML_AVAILABLE = True
except Exception as e:
    print(f"⚠️ ML utilities not available: {e}")
    ML_AVAILABLE = False
    _label_encoder = None

def generate_api_integrated_response(user_message):
    """Generate responses with full API integration"""
    user_message_lower = user_message.lower()
    
    # Clinical Trials queries
    if any(keyword in user_message_lower for keyword in ["trial", "clinical trial", "study", "research"]):
        return handle_clinical_trials_query(user_message)
    
    # CoWIN/Vaccination queries
    elif any(keyword in user_message_lower for keyword in ["cowin", "vaccine", "vaccination", "immunization"]):
        return handle_cowin_query(user_message)
    
    # COVID-19 queries
    elif any(keyword in user_message_lower for keyword in ["covid", "coronavirus", "mohfw", "pandemic"]):
        return handle_covid_query(user_message)
    
    # Medical definition queries
    elif any(keyword in user_message_lower for keyword in ["define", "what is", "meaning of", "definition"]):
        return handle_medical_definition_query(user_message)
    
    # Greetings
    elif any(keyword in user_message_lower for keyword in ["hello", "hi", "hey", "good morning", "good afternoon"]):
        return handle_greeting_query(user_message)
    
    # Symptom analysis
    else:
        return analyze_symptoms_with_ml(user_message)

def handle_clinical_trials_query(user_message):
    """Handle clinical trials queries"""
    try:
        if not APIS_AVAILABLE:
            return {
                "type": "clinical_trials",
                "message": "Clinical trials information would be available here with full API integration.",
                "error": "API integration not available",
                "suggestion": "Please try again later or contact support."
            }
        
        # Extract search terms from the message
        search_terms = extract_search_terms(user_message, ["trial", "clinical trial", "study", "research"])
        
        # Search for clinical trials
        result = search_clinical_trials(search_terms)
        
        if result.get("success"):
            trials = result.get("trials", [])
            total_results = result.get("total_results", 0)
            
            if trials:
                # Format the response
                message = f"Found {total_results} clinical trials for '{search_terms}':\n\n"
                
                for i, trial in enumerate(trials[:3]):  # Show top 3
                    message += f"{i+1}. {trial.get('title', 'No title')}\n"
                    message += f"   Condition: {', '.join(trial.get('conditions', ['Unknown']))}\n"
                    message += f"   Phase: {', '.join(trial.get('phase', ['Unknown']))}\n"
                    message += f"   Status: {trial.get('status', 'Unknown')}\n\n"
                
                if total_results > 3:
                    message += f"... and {total_results - 3} more trials available."
                
                return {
                    "type": "clinical_trials",
                    "message": message,
                    "data": {
                        "trials": trials[:5],  # Include full data for top 5
                        "total_results": total_results,
                        "search_query": search_terms
                    },
                    "api_source": result.get("api_source", "ClinicalTrials.gov"),
                    "response_time": result.get("response_time", 0)
                }
            else:
                return {
                    "type": "clinical_trials",
                    "message": f"No clinical trials found for '{search_terms}'. Try different search terms or check spelling.",
                    "search_query": search_terms,
                    "api_source": result.get("api_source", "ClinicalTrials.gov")
                }
        else:
            return {
                "type": "clinical_trials",
                "message": "Unable to fetch clinical trials information at the moment. Please try again later.",
                "error": result.get("error", "Unknown error"),
                "search_query": search_terms
            }
    
    except Exception as e:
        return {
            "type": "error",
            "message": "I encountered an error while searching for clinical trials. Please try again.",
            "error": str(e)
        }

def handle_cowin_query(user_message):
    """Handle CoWIN/vaccination queries"""
    try:
        if not APIS_AVAILABLE:
            return {
                "type": "cowin",
                "message": "Vaccination information would be available here with full API integration.",
                "error": "API integration not available"
            }
        
        # Extract state if mentioned
        state = extract_state_from_message(user_message)
        
        # Get vaccination statistics
        result = get_cowin_stats(state)
        
        if result.get("success"):
            data = result.get("data", {})
            
            if data.get("type") == "age_wise_vaccination":
                message = "📊 Vaccination Statistics by Age Group:\n\n"
                for age_group in data.get("age_groups", [])[:5]:
                    message += f"Age {age_group.get('age_group', 'Unknown')}:\n"
                    message += f"  • Dose 1: {age_group.get('doses_1', 0):,}\n"
                    message += f"  • Dose 2: {age_group.get('doses_2', 0):,}\n"
                    message += f"  • Total: {age_group.get('total', 0):,}\n\n"
                
                message += f"Total Doses Administered: {data.get('total_doses', 0):,}"
            
            elif data.get("type") == "state_wise_vaccination":
                message = f"📊 State-wise Vaccination Statistics:\n\n"
                message += f"Total Vaccinations: {data.get('total_vaccinations', 0):,}\n"
                message += f"Total Dose 1: {data.get('total_doses_1', 0):,}\n"
                message += f"Total Dose 2: {data.get('total_doses_2', 0):,}\n"
                message += f"Last Updated: {data.get('last_updated', 'Unknown')}"
            
            else:
                message = "📊 Vaccination Statistics Available\n\nCheck the detailed data section for comprehensive information."
            
            return {
                "type": "cowin",
                "message": message,
                "data": data,
                "api_source": result.get("api_source", "CoWIN API"),
                "response_time": result.get("response_time", 0)
            }
        else:
            return {
                "type": "cowin",
                "message": "Unable to fetch vaccination information at the moment. Please try again later.",
                "error": result.get("error", "Unknown error")
            }
    
    except Exception as e:
        return {
            "type": "error",
            "message": "I encountered an error while fetching vaccination data. Please try again.",
            "error": str(e)
        }

def handle_covid_query(user_message):
    """Handle COVID-19 queries"""
    try:
        if not APIS_AVAILABLE:
            return {
                "type": "covid",
                "message": "COVID-19 information would be available here with full API integration.",
                "error": "API integration not available"
            }
        
        # Extract state if mentioned
        state = extract_state_from_message(user_message)
        
        # Get COVID-19 statistics
        result = get_mohfw_data(state)
        
        if result.get("success"):
            data = result.get("data", {})
            
            if state and data.get("type") == "state_data":
                state_info = data.get("state", {})
                message = f"🦠 COVID-19 Statistics for {state}:\n\n"
                message += f"Confirmed: {state_info.get('confirmed', 0):,}\n"
                message += f"Active: {state_info.get('active', 0):,}\n"
                message += f"Recovered: {state_info.get('recovered', 0):,}\n"
                message += f"Deaths: {state_info.get('deaths', 0):,}\n"
                message += f"Last Updated: {state_info.get('lastupdatedtime', 'Unknown')}"
            
            elif data.get("type") == "all_states":
                summary = data.get("total_summary", {})
                message = f"🦠 National COVID-19 Statistics:\n\n"
                message += f"Total Confirmed: {summary.get('confirmed', 0):,}\n"
                message += f"Total Active: {summary.get('active', 0):,}\n"
                message += f"Total Recovered: {summary.get('recovered', 0):,}\n"
                message += f"Total Deaths: {summary.get('deaths', 0):,}\n"
                message += f"Last Updated: {data.get('last_updated', 'Unknown')}\n\n"
                message += f"Data available for {data.get('total_states', 0)} states/UTs"
            
            else:
                message = "🦠 COVID-19 Statistics Available\n\nCheck the detailed data section for comprehensive information."
            
            return {
                "type": "covid",
                "message": message,
                "data": data,
                "api_source": result.get("api_source", "MoHFW API"),
                "response_time": result.get("response_time", 0)
            }
        else:
            return {
                "type": "covid",
                "message": "Unable to fetch COVID-19 information at the moment. Please try again later.",
                "error": result.get("error", "Unknown error")
            }
    
    except Exception as e:
        return {
            "type": "error",
            "message": "I encountered an error while fetching COVID-19 data. Please try again.",
            "error": str(e)
        }

def handle_medical_definition_query(user_message):
    """Handle medical definition queries"""
    try:
        if not APIS_AVAILABLE:
            return {
                "type": "umls",
                "message": "Medical definition would be available here with UMLS integration.",
                "error": "API integration not available"
            }
        
        # Extract medical term from the message
        term = extract_medical_term(user_message)
        
        if not term:
            return {
                "type": "umls",
                "message": "Please specify which medical term you'd like me to define.",
                "suggestion": "Try: 'Define hypertension' or 'What is diabetes?'"
            }
        
        # Get medical definition
        result = get_umls_info(term)
        
        if result.get("success"):
            definition_data = result.get("definition", {})
            
            message = f"📖 Medical Definition for '{term}':\n\n"
            message += f"Definition: {definition_data.get('definition', 'No definition available')}\n"
            
            if definition_data.get("synonyms"):
                message += f"\nSynonyms: {', '.join(definition_data['synonyms'][:3])}"
            
            if definition_data.get("cui"):
                message += f"\nMedical Code: {definition_data['cui']}"
            
            return {
                "type": "umls",
                "message": message,
                "data": {
                    "term": term,
                    "definition": definition_data,
                    "search_query": term
                },
                "api_source": result.get("api_source", "UMLS API"),
                "response_time": result.get("response_time", 0)
            }
        else:
            return {
                "type": "umls",
                "message": f"I couldn't find a definition for '{term}'. Please check the spelling or try a different medical term.",
                "error": result.get("error", "Term not found"),
                "search_query": term,
                "suggestion": "Available terms: hypertension, diabetes, fever, cough, headache, nausea, vomiting, rash"
            }
    
    except Exception as e:
        return {
            "type": "error",
            "message": "I encountered an error while looking up the medical definition. Please try again.",
            "error": str(e)
        }

def handle_greeting_query(user_message):
    """Handle greeting queries"""
    return {
        "type": "greeting",
        "message": "Hello! I'm your AI-powered health assistant with full API integration. I can help you with:\n\n"
                  "🔬 Clinical trials information\n"
                  "💉 Vaccination statistics (CoWIN)\n"
                  "🦠 COVID-19 data (MoHFW)\n"
                  "📖 Medical definitions (UMLS)\n"
                  "🤖 Symptom analysis with ML\n\n"
                  "How can I help you today?"
    }

def analyze_symptoms_with_ml(user_message):
    """Analyze symptoms using the trained ML model"""
    try:
        if ML_AVAILABLE:
            # Use the trained model for prediction
            label, confidence, probabilities = predict_new(user_message)
            
            # Determine confidence level
            if confidence >= 0.8:
                confidence_level = "high"
            elif confidence >= 0.6:
                confidence_level = "medium"
            else:
                confidence_level = "low"
            
            # Get top 3 predictions
            if len(probabilities) > 1:
                # Sort probabilities and get top predictions
                sorted_indices = np.argsort(probabilities)[::-1]
                alternative_conditions = []
                for i in range(1, min(4, len(sorted_indices))):
                    alt_idx = sorted_indices[i]
                    alt_label = _label_encoder.inverse_transform([alt_idx])[0]
                    alt_conf = probabilities[alt_idx]
                    if alt_conf > 0.1:  # Only include if confidence > 10%
                        alternative_conditions.append({
                            'condition': alt_label,
                            'confidence': float(alt_conf),
                            'description': f'Alternative diagnosis with {alt_conf:.1%} confidence'
                        })
            else:
                alternative_conditions = []
            
            # Process text for symptom detection
            processed_text = simple_tokenize(user_message, detect_lang(user_message))
            
            return {
                "type": "diagnosis",
                "disease": label,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "symptoms_detected": extract_symptoms(user_message),
                "alternative_conditions": alternative_conditions,
                "message": f"Based on your symptoms, I suggest: {label} (Confidence: {confidence:.1%})",
                "disclaimer": "This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.",
                "model_info": {
                    "model_type": "Trained Random Forest with TF-IDF",
                    "version": "1.0.0",
                    "available": ML_AVAILABLE,
                    "processed_text": processed_text,
                    "training_data": "1000 synthetic health records"
                }
            }
        else:
            # Fallback to rule-based system
            return analyze_symptoms_fallback(user_message)
    
    except Exception as e:
        print(f"Error in ML prediction: {e}")
        return analyze_symptoms_fallback(user_message)

def analyze_symptoms_fallback(user_message):
    """Fallback rule-based symptom analysis"""
    processed_text = user_message.lower()
    
    # Enhanced symptom detection
    symptoms = []
    
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
    conditions = []
    
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
            "model_type": "Rule-Based Fallback System",
            "version": "1.0.0",
            "available": False,
            "processed_text": processed_text
        }
    }

def extract_symptoms(text):
    """Extract symptoms from text"""
    text_lower = text.lower()
    symptoms = []
    
    symptom_keywords = {
        'fever': ['fever', 'temperature', 'hot', 'burning', 'pyrexia'],
        'cough': ['cough', 'hack', 'throat', 'breathing', 'respiratory'],
        'headache': ['headache', 'head', 'pain', 'cephalalgia'],
        'nausea': ['nausea', 'vomiting', 'stomach', 'digestion', 'queasiness'],
        'rash': ['rash', 'skin', 'eruption', 'spots'],
        'pain': ['pain', 'ache', 'discomfort', 'sore']
    }
    
    for symptom, keywords in symptom_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            symptoms.append(symptom)
    
    return symptoms

def extract_search_terms(user_message, keywords):
    """Extract search terms from user message"""
    # Remove keywords and get the remaining text
    terms = user_message.lower()
    for keyword in keywords:
        terms = terms.replace(keyword, "")
    
    return terms.strip()

def extract_state_from_message(user_message):
    """Extract state name from user message"""
    # Simple state extraction - can be enhanced
    user_message_lower = user_message.lower()
    
    # Common state names
    states = ["maharashtra", "karnataka", "tamil nadu", "kerala", "delhi", "gujarat", "rajasthan", 
              "uttar pradesh", "west bengal", "andhra pradesh", "telangana", "haryana", "punjab"]
    
    for state in states:
        if state in user_message_lower:
            return state
    
    return None

def extract_medical_term(user_message):
    """Extract medical term from user message"""
    # Remove definition keywords
    terms = user_message.lower()
    keywords_to_remove = ["define", "what is", "meaning of", "definition", "tell me about"]
    
    for keyword in keywords_to_remove:
        terms = terms.replace(keyword, "")
    
    return terms.strip()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0-full-api-integration",
        "timestamp": "2024-01-01T00:00:00Z",
        "mode": "Full API Integration with ML",
        "apis_available": APIS_AVAILABLE,
        "ml_available": ML_AVAILABLE,
        "integrated_apis": [
            "ClinicalTrials.gov",
            "CoWIN API", 
            "MoHFW COVID-19 API",
            "UMLS Medical Dictionary"
        ]
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint with full API integration"""
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
        
        # Generate API-integrated bot response
        response = generate_api_integrated_response(user_message)
        
        # Log the interaction
        print(f"User {user_id}: {user_message}")
        print(f"Bot response: {response.get('type', 'unknown')}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "Sorry, I encountered an error. Please try again."
        }), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """Get API status information"""
    return jsonify({
        "apis_available": APIS_AVAILABLE,
        "ml_available": ML_AVAILABLE,
        "integrated_services": {
            "clinical_trials": "ClinicalTrials.gov",
            "cowin": "CoWIN API",
            "covid": "MoHFW COVID-19 API", 
            "umls": "UMLS Medical Dictionary"
        }
    })

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
    print("🚀 Starting HealthBot AI Chatbot with Full API Integration...")
    print("=" * 70)
    
    if APIS_AVAILABLE:
        print("✅ All health APIs integrated successfully!")
    else:
        print("⚠️  Some APIs not available, using fallback mode")
    
    if ML_AVAILABLE:
        print("✅ Bio-ClinicalBERT ML processing available!")
    else:
        print("⚠️  ML processing not available, using enhanced rules")
    
    print("🌐 Web API: http://localhost:5004")
    print("❤️  Health Check: http://localhost:5004/health")
    print("💬 Chat API: POST http://localhost:5004/chat")
    print("📊 API Status: http://localhost:5004/api/status")
    print("=" * 70)
    print("🔬 ClinicalTrials.gov - Search clinical trials")
    print("💉 CoWIN API - Vaccination statistics")
    print("🦠 MoHFW API - COVID-19 data")
    print("📖 UMLS API - Medical definitions")
    print("🤖 ML Model - Symptom analysis")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5004, debug=True)
