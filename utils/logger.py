"""
Logging configuration for HealthBot AI Chatbot
"""
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional
from config.settings import config

class HealthBotLogger:
    """Centralized logging for the HealthBot application"""
    
    def __init__(self, name: str = "healthbot"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self):
        """Setup logger with file and console handlers"""
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        log_level = getattr(logging, config.app.LOG_LEVEL.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(config.app.LOG_FILE), exist_ok=True)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            config.app.LOG_FILE,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger"""
        return self.logger

# Global logger instance
_logger_instance = None

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance"""
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = HealthBotLogger(name or "healthbot")
    
    return _logger_instance.get_logger()

def log_user_interaction(user_id: str, message: str, response: str, 
                        confidence: Optional[float] = None, 
                        prediction: Optional[str] = None):
    """Log user interactions for analytics"""
    logger = get_logger("user_interaction")
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "user_message": message,
        "bot_response": response,
        "confidence": confidence,
        "prediction": prediction
    }
    
    logger.info(f"User Interaction: {log_data}")

def log_api_call(api_name: str, endpoint: str, status_code: int, 
                response_time: float, error: Optional[str] = None):
    """Log API calls for monitoring"""
    logger = get_logger("api_calls")
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "api": api_name,
        "endpoint": endpoint,
        "status_code": status_code,
        "response_time": response_time,
        "error": error
    }
    
    if status_code >= 400:
        logger.error(f"API Error: {log_data}")
    else:
        logger.info(f"API Call: {log_data}")

def log_model_prediction(input_text: str, prediction: str, confidence: float, 
                        model_version: str = "latest"):
    """Log ML model predictions"""
    logger = get_logger("model_predictions")
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "input_text": input_text,
        "prediction": prediction,
        "confidence": confidence,
        "model_version": model_version
    }
    
    logger.info(f"Model Prediction: {log_data}")

# Initialize logger when module is imported
logger = get_logger()

