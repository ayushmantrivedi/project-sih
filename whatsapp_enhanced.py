"""
Enhanced WhatsApp integration with rich media support and better formatting
"""
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from chatbot import generate_bot_response
from config import get_config
from utils import get_logger
import json
import re

app = Flask(__name__)
config = get_config()
logger = get_logger("whatsapp_enhanced")

# Initialize Twilio client
twilio_client = Client(config.twilio.ACCOUNT_SID, config.twilio.AUTH_TOKEN)

def format_whatsapp_response(response_data, user_message):
    """Format bot response for WhatsApp with rich formatting"""
    
    if response_data.get('type') == 'diagnosis':
        disease = response_data.get('disease', 'Unknown')
        confidence = response_data.get('confidence', 0)
        confidence_level = response_data.get('confidence_level', 'low')
        
        # Create formatted response
        reply = f"🏥 *Health Analysis*\n\n"
        reply += f"*Predicted Condition:* {disease}\n"
        reply += f"*Confidence Level:* {confidence:.1%} ({confidence_level})\n\n"
        
        # Add confidence-based advice
        if confidence_level == 'high':
            reply += "✅ This prediction has high confidence. However, please consult a healthcare professional for proper diagnosis.\n\n"
        elif confidence_level == 'medium':
            reply += "⚠️ This prediction has medium confidence. It's recommended to consult a healthcare professional.\n\n"
        else:
            reply += "❌ This prediction has low confidence. Please consult a healthcare professional immediately.\n\n"
        
        # Add disclaimer
        reply += "⚠️ *Disclaimer:* This is for informational purposes only and should not replace professional medical advice."
        
        return reply
        
    elif response_data.get('type') == 'covid':
        content = response_data.get('content', {})
        reply = f"📊 *COVID-19 Statistics*\n\n"
        
        if isinstance(content, dict):
            for key, value in content.items():
                reply += f"*{key.replace('_', ' ').title()}:* {value}\n"
        else:
            reply += str(content)
            
        return reply
        
    elif response_data.get('type') == 'cowin':
        content = response_data.get('content', {})
        reply = f"💉 *Vaccination Statistics*\n\n"
        
        if isinstance(content, dict):
            for key, value in content.items():
                reply += f"*{key.replace('_', ' ').title()}:* {value}\n"
        else:
            reply += str(content)
            
        return reply
        
    elif response_data.get('type') == 'clinical_trials':
        content = response_data.get('content', [])
        reply = f"🔬 *Clinical Trials*\n\n"
        
        if isinstance(content, list) and content:
            for i, trial in enumerate(content[:3], 1):  # Limit to 3 trials
                if isinstance(trial, dict):
                    title = trial.get('title', 'Unknown Trial')
                    status = trial.get('status', 'Unknown')
                    reply += f"{i}. *{title}*\n   Status: {status}\n\n"
        else:
            reply += "No clinical trials found for your query."
            
        return reply
        
    elif response_data.get('type') == 'umls':
        content = response_data.get('content', {})
        reply = f"📚 *Medical Definition*\n\n"
        
        if isinstance(content, dict):
            term = content.get('term', 'Unknown')
            definition = content.get('definition', 'No definition available')
            reply += f"*Term:* {term}\n\n"
            reply += f"*Definition:* {definition}"
        else:
            reply += str(content)
            
        return reply
        
    else:
        # Generic error or unknown response
        return f"🤖 *HealthBot Response*\n\n{str(response_data.get('content', 'Sorry, I could not process your request.'))}"

def detect_message_type(message):
    """Detect the type of message (text, image, document, etc.)"""
    # This would be enhanced to detect media types from Twilio webhook
    return 'text'

def handle_media_message(media_url, message_type):
    """Handle media messages (images, documents)"""
    # Future enhancement: Process medical images, lab reports, etc.
    return "I received your media file. Currently, I can only process text messages. Please describe your symptoms in text."

@app.route('/webhook/whatsapp', methods=['POST'])
def whatsapp_webhook():
    """Enhanced WhatsApp webhook with rich formatting"""
    try:
        # Get message details
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        to_number = request.values.get('To', '')
        message_sid = request.values.get('MessageSid', '')
        
        # Check for media messages
        num_media = int(request.values.get('NumMedia', 0))
        
        logger.info(f"WhatsApp message from {from_number}: {incoming_msg}")
        
        if not incoming_msg and num_media == 0:
            return "No message received", 400
        
        # Handle media messages
        if num_media > 0:
            media_url = request.values.get('MediaUrl0', '')
            media_type = request.values.get('MediaContentType0', '')
            reply = handle_media_message(media_url, media_type)
        else:
            # Process text message
            response_data = generate_bot_response(incoming_msg)
            reply = format_whatsapp_response(response_data, incoming_msg)
        
        # Create Twilio response
        resp = MessagingResponse()
        resp.message(reply)
        
        # Log the interaction
        from utils import log_user_interaction
        log_user_interaction(
            user_id=from_number,
            message=incoming_msg,
            response=reply,
            confidence=response_data.get('confidence') if 'response_data' in locals() else None,
            prediction=response_data.get('disease') if 'response_data' in locals() else None,
            platform="whatsapp"
        )
        
        return str(resp)
        
    except Exception as e:
        logger.error(f"Error in WhatsApp webhook: {str(e)}")
        resp = MessagingResponse()
        resp.message("🤖 Sorry, I encountered an error. Please try again or contact support.")
        return str(resp)

@app.route('/whatsapp/status', methods=['POST'])
def whatsapp_status():
    """Handle WhatsApp message status updates"""
    try:
        message_sid = request.values.get('MessageSid', '')
        status = request.values.get('MessageStatus', '')
        
        logger.info(f"WhatsApp message {message_sid} status: {status}")
        
        # You can implement status tracking here
        # e.g., mark messages as delivered, failed, etc.
        
        return "OK", 200
        
    except Exception as e:
        logger.error(f"Error in WhatsApp status webhook: {str(e)}")
        return "Error", 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)