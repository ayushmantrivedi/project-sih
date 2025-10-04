from apis.clinicaltrials import get_clinical_trials
from apis.cowin import get_cowin_stats
from apis.mohfw import get_mohfw_data
from apis.umls import get_umls_info
from models.ml_predict import predict_disease
from config import get_config
from utils import get_logger, log_api_call

def generate_bot_response(user_message):
    """
    Generate bot response based on user message
    """
    config = get_config()
    logger = get_logger("chatbot")
    
    try:
        user_message_lower = user_message.lower()
        
        # Check for specific API queries
        if "trial" in user_message_lower or "clinical trial" in user_message_lower:
            try:
                trials = get_clinical_trials(user_message)
                return {"type": "clinical_trials", "content": trials}
            except Exception as e:
                logger.error(f"Error fetching clinical trials: {str(e)}")
                return {"type": "error", "content": "Unable to fetch clinical trial information at the moment."}
                
        elif "cowin" in user_message_lower or "vaccine" in user_message_lower or "vaccination" in user_message_lower:
            try:
                stats = get_cowin_stats()
                return {"type": "cowin", "content": stats}
            except Exception as e:
                logger.error(f"Error fetching CoWIN data: {str(e)}")
                return {"type": "error", "content": "Unable to fetch vaccination information at the moment."}
                
        elif "covid" in user_message_lower or "mohfw" in user_message_lower or "coronavirus" in user_message_lower:
            try:
                stats = get_mohfw_data()
                return {"type": "covid", "content": stats}
            except Exception as e:
                logger.error(f"Error fetching MoHFW data: {str(e)}")
                return {"type": "error", "content": "Unable to fetch COVID-19 information at the moment."}
                
        elif "define" in user_message_lower or "what is" in user_message_lower:
            try:
                term = user_message.replace("define", "").replace("what is", "").strip()
                info = get_umls_info(term)
                return {"type": "umls", "content": info}
            except Exception as e:
                logger.error(f"Error fetching UMLS info: {str(e)}")
                return {"type": "error", "content": "Unable to fetch medical definition at the moment."}
                
        else:
            # ML prediction for symptoms/disease
            try:
                label, conf, probs = predict_disease(user_message)
                
                # Determine confidence level and format response accordingly
                if conf >= config.ml.HIGH_CONFIDENCE_THRESHOLD:
                    confidence_level = "high"
                elif conf >= config.ml.MEDIUM_CONFIDENCE_THRESHOLD:
                    confidence_level = "medium"
                else:
                    confidence_level = "low"
                
                response = {
                    "type": "diagnosis",
                    "disease": label,
                    "confidence": conf,
                    "confidence_level": confidence_level,
                    "probabilities": probs.tolist() if hasattr(probs, 'tolist') else probs
                }
                
                # Add disclaimer for low confidence predictions
                if confidence_level == "low":
                    response["disclaimer"] = "This is a low-confidence prediction. Please consult a healthcare professional."
                
                return response
                
            except Exception as e:
                logger.error(f"Error in ML prediction: {str(e)}")
                return {
                    "type": "error", 
                    "content": "I'm having trouble analyzing your symptoms right now. Please try again or consult a healthcare professional."
                }
                
    except Exception as e:
        logger.error(f"Unexpected error in generate_bot_response: {str(e)}")
        return {
            "type": "error",
            "content": "I encountered an unexpected error. Please try again."
        }