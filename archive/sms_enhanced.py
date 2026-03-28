"""
Enhanced SMS integration with proper message truncation and formatting
"""
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from chatbot import generate_bot_response
from config import get_config
from utils import get_logger
import re

app = Flask(__name__)
config = get_config()
logger = get_logger("sms_enhanced")

# Initialize Twilio client
twilio_client = Client(config.twilio.ACCOUNT_SID, config.twilio.AUTH_TOKEN)

def format_sms_response(response_data, user_message):
    """Format bot response for SMS with proper truncation"""
    
    if response_data.get('type') == 'diagnosis':
        disease = response_data.get('disease', 'Unknown')
        confidence = response_data.get('confidence', 0)
        confidence_level = response_data.get('confidence_level', 'low')
        
        # Create concise SMS response
        reply = f"HealthBot: {disease} ({confidence:.0%})"
        
        # Add confidence indicator
        if confidence_level == 'high':
            reply += " - High confidence"
        elif confidence_level == 'medium':
            reply += " - Medium confidence"
        else:
            reply += " - Low confidence, consult doctor"
        
        # Add disclaimer (shortened)
        reply += ". Not medical advice."
        
        return reply
        
    elif response_data.get('type') == 'covid':
        content = response_data.get('content', {})
        reply = "COVID Stats: "
        
        if isinstance(content, dict):
            # Get key stats only
            key_stats = []
            for key in ['total_cases', 'active_cases', 'recovered', 'deaths']:
                if key in content:
                    key_stats.append(f"{key.replace('_', ' ').title()}: {content[key]}")
            
            if key_stats:
                reply += ", ".join(key_stats[:2])  # Limit to 2 stats
            else:
                reply += "Data available"
        else:
            reply += str(content)[:50]  # Truncate long content
            
        return reply
        
    elif response_data.get('type') == 'cowin':
        content = response_data.get('content', {})
        reply = "Vaccination: "
        
        if isinstance(content, dict):
            # Get key vaccination stats
            key_stats = []
            for key in ['total_vaccinations', 'first_dose', 'second_dose']:
                if key in content:
                    key_stats.append(f"{key.replace('_', ' ').title()}: {content[key]}")
            
            if key_stats:
                reply += ", ".join(key_stats[:2])  # Limit to 2 stats
            else:
                reply += "Data available"
        else:
            reply += str(content)[:50]  # Truncate long content
            
        return reply
        
    elif response_data.get('type') == 'clinical_trials':
        content = response_data.get('content', [])
        reply = "Clinical Trials: "
        
        if isinstance(content, list) and content:
            trial_count = len(content)
            reply += f"{trial_count} trials found"
        else:
            reply += "No trials found"
            
        return reply
        
    elif response_data.get('type') == 'umls':
        content = response_data.get('content', {})
        reply = "Medical Term: "
        
        if isinstance(content, dict):
            term = content.get('term', 'Unknown')
            definition = content.get('definition', 'No definition')
            # Truncate definition
            if len(definition) > 50:
                definition = definition[:47] + "..."
            reply += f"{term} - {definition}"
        else:
            reply += str(content)[:50]
            
        return reply
        
    else:
        # Generic error or unknown response
        content = str(response_data.get('content', 'Sorry, unable to process.'))
        if len(content) > 50:
            content = content[:47] + "..."
        return f"HealthBot: {content}"

def truncate_sms_message(message, max_length=160):
    """Truncate message to fit SMS limits"""
    if len(message) <= max_length:
        return message
    
    # Try to truncate at word boundary
    truncated = message[:max_length-3]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can truncate at a reasonable word boundary
        return truncated[:last_space] + "..."
    else:
        return truncated + "..."

def send_multipart_sms(to_number, message, from_number=None):
    """Send multipart SMS for long messages"""
    if from_number is None:
        from_number = config.twilio.SMS_NUMBER
    
    # Split message into parts if too long
    max_length = 160
    if len(message) <= max_length:
        return twilio_client.messages.create(
            body=message,
            from_=from_number,
            to=to_number
        )
    
    # Split into multiple messages
    parts = []
    current_part = ""
    
    for word in message.split():
        if len(current_part + " " + word) <= max_length:
            current_part += " " + word if current_part else word
        else:
            if current_part:
                parts.append(current_part)
            current_part = word
    
    if current_part:
        parts.append(current_part)
    
    # Send each part
    messages = []
    for i, part in enumerate(parts, 1):
        if len(parts) > 1:
            part = f"({i}/{len(parts)}) {part}"
        
        msg = twilio_client.messages.create(
            body=part,
            from_=from_number,
            to=to_number
        )
        messages.append(msg)
    
    return messages

@app.route('/webhook/sms', methods=['POST'])
def sms_webhook():
    """Enhanced SMS webhook with proper formatting and truncation"""
    try:
        # Get message details
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        to_number = request.values.get('To', '')
        message_sid = request.values.get('MessageSid', '')
        
        logger.info(f"SMS from {from_number}: {incoming_msg}")
        
        if not incoming_msg:
            return "No message received", 400
        
        # Process message
        response_data = generate_bot_response(incoming_msg)
        reply = format_sms_response(response_data, incoming_msg)
        
        # Truncate if necessary
        reply = truncate_sms_message(reply, 160)
        
        # Create Twilio response
        resp = MessagingResponse()
        resp.message(reply)
        
        # Log the interaction
        from utils import log_user_interaction
        log_user_interaction(
            user_id=from_number,
            message=incoming_msg,
            response=reply,
            confidence=response_data.get('confidence'),
            prediction=response_data.get('disease'),
            platform="sms"
        )
        
        return str(resp)
        
    except Exception as e:
        logger.error(f"Error in SMS webhook: {str(e)}")
        resp = MessagingResponse()
        resp.message("Error occurred. Please try again.")
        return str(resp)

@app.route('/sms/send', methods=['POST'])
def send_sms():
    """Send SMS programmatically"""
    try:
        data = request.get_json()
        to_number = data.get('to')
        message = data.get('message')
        
        if not to_number or not message:
            return jsonify({"error": "Missing 'to' or 'message' field"}), 400
        
        # Send SMS
        result = send_multipart_sms(to_number, message)
        
        return jsonify({
            "success": True,
            "message_sid": result.sid if hasattr(result, 'sid') else [msg.sid for msg in result],
            "status": "sent"
        })
        
    except Exception as e:
        logger.error(f"Error sending SMS: {str(e)}")
        return jsonify({"error": "Failed to send SMS"}), 500

@app.route('/sms/status', methods=['POST'])
def sms_status():
    """Handle SMS status updates"""
    try:
        message_sid = request.values.get('MessageSid', '')
        status = request.values.get('MessageStatus', '')
        
        logger.info(f"SMS {message_sid} status: {status}")
        
        # You can implement status tracking here
        # e.g., mark messages as delivered, failed, etc.
        
        return "OK", 200
        
    except Exception as e:
        logger.error(f"Error in SMS status webhook: {str(e)}")
        return "Error", 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)