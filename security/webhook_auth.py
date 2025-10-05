"""
Webhook security and authentication for WhatsApp and SMS
"""
import hmac
import hashlib
import base64
from flask import request, abort
from config import get_config
from utils import get_logger
import time

config = get_config()
logger = get_logger("webhook_auth")

def verify_twilio_signature(request_signature, url, params):
    """Verify Twilio webhook signature"""
    try:
        # Get Twilio auth token
        auth_token = config.twilio.AUTH_TOKEN
        
        if not auth_token:
            logger.error("Twilio auth token not configured")
            return False
        
        # Create expected signature
        signature = create_twilio_signature(url, params, auth_token)
        
        # Compare signatures
        return hmac.compare_digest(signature, request_signature)
        
    except Exception as e:
        logger.error(f"Error verifying Twilio signature: {str(e)}")
        return False

def create_twilio_signature(url, params, auth_token):
    """Create Twilio signature for verification"""
    # Sort parameters alphabetically
    sorted_params = sorted(params.items())
    
    # Create query string
    query_string = '&'.join([f"{key}={value}" for key, value in sorted_params])
    
    # Create signature string
    signature_string = url + query_string
    
    # Create HMAC signature
    signature = hmac.new(
        auth_token.encode('utf-8'),
        signature_string.encode('utf-8'),
        hashlib.sha1
    ).digest()
    
    # Encode as base64
    return base64.b64encode(signature).decode('utf-8')

def verify_webhook_request():
    """Verify webhook request authenticity"""
    try:
        # Get signature from headers
        signature = request.headers.get('X-Twilio-Signature')
        
        if not signature:
            logger.warning("No Twilio signature found in request")
            return False
        
        # Get full URL
        url = request.url
        
        # Get form data
        params = request.form.to_dict()
        
        # Verify signature
        is_valid = verify_twilio_signature(signature, url, params)
        
        if not is_valid:
            logger.warning("Invalid Twilio signature")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying webhook request: {str(e)}")
        return False

def require_twilio_auth(f):
    """Decorator to require Twilio authentication"""
    def decorated_function(*args, **kwargs):
        if not verify_webhook_request():
            logger.warning(f"Unauthorized webhook request from {request.remote_addr}")
            abort(403)
        return f(*args, **kwargs)
    
    decorated_function.__name__ = f.__name__
    return decorated_function

def rate_limit_by_phone(phone_number, max_requests=10, window_minutes=5):
    """Rate limit by phone number"""
    # This is a simple in-memory rate limiting
    # In production, use Redis or database for persistence
    
    current_time = time.time()
    window_seconds = window_minutes * 60
    
    # Clean old entries (simple cleanup)
    if not hasattr(rate_limit_by_phone, 'requests'):
        rate_limit_by_phone.requests = {}
    
    # Remove old entries
    rate_limit_by_phone.requests = {
        phone: timestamps for phone, timestamps in rate_limit_by_phone.requests.items()
        if any(t > current_time - window_seconds for t in timestamps)
    }
    
    # Check if phone number exists
    if phone_number not in rate_limit_by_phone.requests:
        rate_limit_by_phone.requests[phone_number] = []
    
    # Add current request
    rate_limit_by_phone.requests[phone_number].append(current_time)
    
    # Check if over limit
    recent_requests = [
        t for t in rate_limit_by_phone.requests[phone_number]
        if t > current_time - window_seconds
    ]
    
    if len(recent_requests) > max_requests:
        logger.warning(f"Rate limit exceeded for phone {phone_number}")
        return False
    
    return True

def validate_phone_number(phone_number):
    """Validate phone number format"""
    if not phone_number:
        return False
    
    # Remove non-digit characters
    digits_only = ''.join(filter(str.isdigit, phone_number))
    
    # Check length (basic validation)
    if len(digits_only) < 10 or len(digits_only) > 15:
        return False
    
    return True

def sanitize_input(text):
    """Sanitize user input"""
    if not text:
        return ""
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', '\x00']
    for char in dangerous_chars:
        text = text.replace(char, '')
    
    # Limit length
    if len(text) > 1000:
        text = text[:1000]
    
    return text.strip()

def log_security_event(event_type, phone_number, details=None):
    """Log security events"""
    logger.warning(f"Security event: {event_type} from {phone_number} - {details or 'No details'}")

def check_blacklist(phone_number):
    """Check if phone number is blacklisted"""
    # In production, this would check against a database
    blacklisted_numbers = [
        # Add blacklisted numbers here
    ]
    
    return phone_number in blacklisted_numbers

def security_headers():
    """Add security headers to response"""
    from flask import make_response
    
    response = make_response()
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    return response