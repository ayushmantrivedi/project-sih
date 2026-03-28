"""
Configuration settings for HealthBot AI Chatbot
"""
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class APIConfig:
    """API configuration settings"""
    # ClinicalTrials API
    CLINICAL_TRIALS_BASE_URL: str = "https://clinicaltrials.gov/api/query/full_studies"
    CLINICAL_TRIALS_MAX_RESULTS: int = 5
    
    # CoWIN API
    COWIN_BASE_URL: str = "https://data.cowin.gov.in/api/v1/reports"
    
    # MoHFW API
    MOHFW_BASE_URL: str = "https://data.covid19india.org/data.json"
    
    # UMLS API (requires license)
    UMLS_BASE_URL: str = "https://uts-ws.nlm.nih.gov/rest"
    UMLS_API_KEY: str = os.getenv("UMLS_API_KEY", "")
    
    # Infermedica API
    INFERMEDICA_BASE_URL: str = "https://api.infermedica.com/v3"
    INFERMEDICA_APP_ID: str = os.getenv("INFERMEDICA_APP_ID", "")
    INFERMEDICA_APP_KEY: str = os.getenv("INFERMEDICA_APP_KEY", "")

@dataclass
class MLConfig:
    """Machine Learning model configuration"""
    MODEL_NAME: str = "emilyalsentzer/Bio_ClinicalBERT"
    MAX_LENGTH: int = 64
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 30
    BASE_LR: float = 2e-5
    SEED: int = 42
    
    # Model paths
    MODEL_WEIGHTS_PATH: str = "models/best_classifier_weights.h5"
    MODEL_BUNDLE_PATH: str = "models/sih_model_bundle.joblib"
    TOKENIZER_PATH: str = "models/saved_tokenizer"
    ENCODER_PATH: str = "models/saved_encoder"
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD: float = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.6
    LOW_CONFIDENCE_THRESHOLD: float = 0.4

@dataclass
class DatabaseConfig:
    """Database configuration"""
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///healthbot.db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Session settings
    SESSION_TIMEOUT: int = 3600  # 1 hour
    MAX_CONVERSATION_HISTORY: int = 50

@dataclass
class TwilioConfig:
    """Twilio configuration for WhatsApp and SMS"""
    ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    WHATSAPP_NUMBER: str = os.getenv("TWILIO_WHATSAPP_NUMBER", "")
    SMS_NUMBER: str = os.getenv("TWILIO_SMS_NUMBER", "")
    
    # Webhook URLs
    WHATSAPP_WEBHOOK_URL: str = os.getenv("WHATSAPP_WEBHOOK_URL", "")
    SMS_WEBHOOK_URL: str = os.getenv("SMS_WEBHOOK_URL", "")

@dataclass
class AppConfig:
    """Application configuration"""
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "5000"))
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 30
    RATE_LIMIT_BURST: int = 10
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = "logs/healthbot.log"
    
    # Caching
    CACHE_TTL: int = 300  # 5 minutes
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "True").lower() == "true"

@dataclass
class HealthBotConfig:
    """Main configuration class"""
    api: APIConfig = None
    ml: MLConfig = None
    database: DatabaseConfig = None
    twilio: TwilioConfig = None
    app: AppConfig = None
    
    # Language settings
    SUPPORTED_LANGUAGES: list = None
    DEFAULT_LANGUAGE: str = "en"
    
    # Response templates
    RESPONSE_TEMPLATES: Dict[str, Any] = None
    
    def __post_init__(self):
        # Initialize configuration objects if they are None
        if self.api is None:
            self.api = APIConfig()
        if self.ml is None:
            self.ml = MLConfig()
        if self.database is None:
            self.database = DatabaseConfig()
        if self.twilio is None:
            self.twilio = TwilioConfig()
        if self.app is None:
            self.app = AppConfig()
        
        if self.SUPPORTED_LANGUAGES is None:
            self.SUPPORTED_LANGUAGES = ["en", "hi"]
        
        if self.RESPONSE_TEMPLATES is None:
            self.RESPONSE_TEMPLATES = {
                "greeting": {
                    "en": "Hello! I'm your health assistant. How can I help you today?",
                    "hi": "नमस्ते! मैं आपकी स्वास्थ्य सहायक हूं। आज मैं आपकी कैसे मदद कर सकती हूं?"
                },
                "symptom_analysis": {
                    "en": "Based on your symptoms, I suggest: {prediction} (Confidence: {confidence:.1%})",
                    "hi": "आपके लक्षणों के आधार पर, मैं सुझाव देती हूं: {prediction} (विश्वसनीयता: {confidence:.1%})"
                },
                "low_confidence": {
                    "en": "I'm not very confident about this prediction. Please consult a healthcare professional.",
                    "hi": "मुझे इस भविष्यवाणी पर ज्यादा भरोसा नहीं है। कृपया एक स्वास्थ्य पेशेवर से परामर्श करें।"
                },
                "error": {
                    "en": "Sorry, I encountered an error. Please try again.",
                    "hi": "क्षमा करें, मुझे एक त्रुटि का सामना करना पड़ा। कृपया पुनः प्रयास करें।"
                }
            }

# Global configuration instance
config = HealthBotConfig()

def get_config() -> HealthBotConfig:
    """Get the global configuration instance"""
    return config

def validate_config() -> bool:
    """Validate configuration settings"""
    warnings = []
    
    # Check required environment variables
    required_vars = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN"
    ]
    
    for var in required_vars:
        if not os.getenv(var) or "your_" in os.getenv(var):
            warnings.append(f"Missing or default environment variable: {var}")
    
    # Check model files (Warning only, since we are moving to Agentic RAG)
    if not os.path.exists(config.ml.MODEL_WEIGHTS_PATH):
        warnings.append(f"Legacy model weights file not found: {config.ml.MODEL_WEIGHTS_PATH}. System will prioritize MediAgent/Ollama logic.")
    
    if warnings:
        print("Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        # We return True to allow the app to run in development mode
        return True
    
    return True

if __name__ == "__main__":
    # Validate configuration when run directly
    if validate_config():
        print("Configuration validation passed!")
    else:
        print("Configuration validation failed!")
