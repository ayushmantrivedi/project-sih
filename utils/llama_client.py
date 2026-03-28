import requests
import json
from config import get_config
from utils import get_logger

logger = get_logger("llama_client")

class LlamaClient:
    """
    Client for interacting with local Llama instance via Ollama.
    """
    def __init__(self, model="phi3:mini", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = f"{base_url}/api/generate"
        
    def generate_reasoning(self, user_message, predicted_disease):
        """
        Generate a medical reasoning/explanation for a given symptom and prediction.
        """
        prompt = f"""
        System: You are a professional medical reasoning assistant. 
        A user has reported symptoms and our model has predicted a potential condition. 
        Your task is to provide a brief, professional explanation of WHY these symptoms might lead to this condition.
        
        Symptoms reported: "{user_message}"
        Predicted Condition: "{predicted_disease}"
        
        Please provide a concise medical reasoning (3-5 sentences). 
        Always include a disclaimer that this is AI-generated and not a substitute for professional medical advice.
        """
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No reasoning could be generated at this time.")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            return "Unable to connect to the reasoning engine. Please ensure Ollama is running."

# Singleton instance
_llama_client = None

def get_llama_client():
    global _llama_client
    if _llama_client is None:
        # You can customize the model name here based on what's available (phi3, llama3, etc.)
        _llama_client = LlamaClient(model="llama3:latest")
    return _llama_client
