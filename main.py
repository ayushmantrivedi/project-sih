"""
Main application entry point for HealthBot AI Chatbot
"""
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config, validate_config
from utils import get_logger
from chatbot import generate_bot_response

# Initialize configuration
config = get_config()
logger = get_logger("main")

# Validate configuration
if not validate_config():
    logger.error("Configuration validation failed!")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config.app.SECRET_KEY

# Enable CORS
CORS(app)

# Initialize rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[f"{config.app.RATE_LIMIT_PER_MINUTE} per minute"],
    storage_uri="memory://"
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": "2024-01-01T00:00:00Z"
    })

@app.route('/chat', methods=['POST'])
@limiter.limit(f"{config.app.RATE_LIMIT_PER_MINUTE} per minute")
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
        
        # Get user ID for tracking (critical for per-user persistent memory)
        user_id = data.get('user_id', 'anonymous')
        
        # Generate bot response using the new Agent Orchestrator
        response_data = generate_bot_response(user_message, user_id=user_id)
        
        # Log the interaction
        from utils import log_user_interaction
        log_user_interaction(
            user_id=user_id,
            message=user_message,
            response=response_data.get('content', ''),
            prediction="MediAgent Analysis"
        )
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": "Sorry, I encountered an error. Please try again."
        }), 500

@app.route('/webhook/whatsapp', methods=['POST'])
@limiter.limit(f"{config.app.RATE_LIMIT_BURST} per minute")
def whatsapp_webhook():
    """WhatsApp webhook endpoint"""
    try:
        from twilio.twiml.messaging_response import MessagingResponse
        
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        
        if not incoming_msg:
            return "No message received", 400
        
        # Use incoming number as user_id for persistent memory
        user_id = from_number
        
        # Generate agentic response
        response_data = generate_bot_response(incoming_msg, user_id=user_id)
        
        # Format response for WhatsApp (Directly use the 'content' or reasoning)
        reply = response_data.get('content', 'Sorry, I could not process your request.')
        
        # Create Twilio response
        resp = MessagingResponse()
        resp.message(reply)
        
        # Log the interaction
        from utils import log_user_interaction
        log_user_interaction(
            user_id=user_id,
            message=incoming_msg,
            response=reply,
            prediction="MediAgent Analysis"
        )
        
        return str(resp)
        
    except Exception as e:
        logger.error(f"Error in WhatsApp webhook: {str(e)}")
        resp = MessagingResponse()
        resp.message("Sorry, I encountered an error. Please try again.")
        return str(resp)

@app.route('/webhook/sms', methods=['POST'])
@limiter.limit(f"{config.app.RATE_LIMIT_BURST} per minute")
def sms_webhook():
    """SMS webhook endpoint"""
    try:
        from twilio.twiml.messaging_response import MessagingResponse
        
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        
        if not incoming_msg:
            return "No message received", 400
        
        # Generate response using phone number as session identifier
        user_id = from_number
        response_data = generate_bot_response(incoming_msg, user_id=user_id)
        
        # Format response for SMS (More concise summaries)
        reply = response_data.get('content', 'Unable to process request.')
        
        # Truncate if too long for SMS
        if len(reply) > 160:
            reply = reply[:157] + "..."
        
        # Create Twilio response
        resp = MessagingResponse()
        resp.message(reply)
        
        # Log the interaction
        from utils import log_user_interaction
        log_user_interaction(
            user_id=user_id,
            message=incoming_msg,
            response=reply,
            prediction="MediAgent Analysis"
        )
        
        return str(resp)
        
    except Exception as e:
        logger.error(f"Error in SMS webhook: {str(e)}")
        resp = MessagingResponse()
        resp.message("Error occurred. Please try again.")
        return str(resp)

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limiting errors"""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please slow down."
    }), 429

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
    logger.info("Starting HealthBot AI Chatbot...")
    logger.info(f"Debug mode: {config.app.DEBUG}")
    logger.info(f"Host: {config.app.HOST}")
    logger.info(f"Port: {config.app.PORT}")
    
    app.run(
        host=config.app.HOST,
        port=config.app.PORT,
        debug=config.app.DEBUG
    )

