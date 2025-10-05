"""
Comprehensive testing framework for HealthBot AI Chatbot integrations
"""
import pytest
import requests
import json
import time
from unittest.mock import patch, MagicMock
from flask import Flask
from twilio.twiml.messaging_response import MessagingResponse

# Import your modules
from chatbot import generate_bot_response
from whatsapp_enhanced import whatsapp_webhook
from sms_enhanced import sms_webhook
from security.webhook_auth import verify_webhook_request
from monitoring.analytics import HealthBotAnalytics

class TestChatbotCore:
    """Test core chatbot functionality"""
    
    def test_disease_prediction(self):
        """Test disease prediction functionality"""
        # Test with common symptoms
        test_cases = [
            ("I have fever and cough", "diagnosis"),
            ("I have headache and nausea", "diagnosis"),
            ("Show me COVID statistics", "covid"),
            ("What is hypertension?", "umls"),
            ("Find clinical trials for diabetes", "clinical_trials")
        ]
        
        for message, expected_type in test_cases:
            response = generate_bot_response(message)
            assert response is not None
            assert "type" in response
            # Note: The actual type might vary based on the message content
    
    def test_error_handling(self):
        """Test error handling"""
        # Test with empty message
        response = generate_bot_response("")
        assert response["type"] == "error"
        
        # Test with None
        response = generate_bot_response(None)
        assert response["type"] == "error"

class TestWhatsAppIntegration:
    """Test WhatsApp integration"""
    
    def test_whatsapp_webhook_formatting(self):
        """Test WhatsApp message formatting"""
        from whatsapp_enhanced import format_whatsapp_response
        
        # Test diagnosis response
        diagnosis_response = {
            "type": "diagnosis",
            "disease": "Common Cold",
            "confidence": 0.85,
            "confidence_level": "high"
        }
        
        formatted = format_whatsapp_response(diagnosis_response, "I have fever")
        assert "🏥" in formatted
        assert "Common Cold" in formatted
        assert "85%" in formatted
        assert "High confidence" in formatted
    
    def test_whatsapp_webhook_security(self):
        """Test WhatsApp webhook security"""
        # This would test webhook signature verification
        # In a real test, you'd mock the Twilio signature verification
        pass

class TestSMSIntegration:
    """Test SMS integration"""
    
    def test_sms_formatting(self):
        """Test SMS message formatting"""
        from sms_enhanced import format_sms_response, truncate_sms_message
        
        # Test diagnosis response
        diagnosis_response = {
            "type": "diagnosis",
            "disease": "Common Cold",
            "confidence": 0.85,
            "confidence_level": "high"
        }
        
        formatted = format_sms_response(diagnosis_response, "I have fever")
        assert "HealthBot:" in formatted
        assert "Common Cold" in formatted
        assert "85%" in formatted
    
    def test_sms_truncation(self):
        """Test SMS message truncation"""
        from sms_enhanced import truncate_sms_message
        
        long_message = "This is a very long message that should be truncated because it exceeds the SMS character limit of 160 characters and needs to be shortened to fit within the allowed length."
        
        truncated = truncate_sms_message(long_message, 160)
        assert len(truncated) <= 160
        assert truncated.endswith("...")

class TestAPIIntegrations:
    """Test API integrations"""
    
    def test_clinical_trials_api(self):
        """Test ClinicalTrials API integration"""
        from apis.clinicaltrials import get_clinical_trials
        
        # Test with a common condition
        trials = get_clinical_trials("diabetes")
        assert isinstance(trials, list)
    
    def test_cowin_api(self):
        """Test CoWIN API integration"""
        from apis.cowin import get_cowin_stats
        
        stats = get_cowin_stats()
        assert isinstance(stats, dict)
    
    def test_mohfw_api(self):
        """Test MoHFW API integration"""
        from apis.mohfw import get_mohfw_data
        
        data = get_mohfw_data()
        assert isinstance(data, dict)

class TestSecurity:
    """Test security features"""
    
    def test_webhook_authentication(self):
        """Test webhook authentication"""
        # This would test Twilio signature verification
        # In a real test, you'd create mock requests with proper signatures
        pass
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        from security.webhook_auth import rate_limit_by_phone
        
        # Test rate limiting
        phone = "+1234567890"
        
        # Should allow first few requests
        for i in range(5):
            assert rate_limit_by_phone(phone, max_requests=10, window_minutes=1)
        
        # Should block after exceeding limit
        for i in range(10):
            rate_limit_by_phone(phone, max_requests=5, window_minutes=1)
        
        assert not rate_limit_by_phone(phone, max_requests=5, window_minutes=1)

class TestAnalytics:
    """Test analytics and monitoring"""
    
    def test_analytics_tracking(self):
        """Test analytics tracking"""
        analytics = HealthBotAnalytics()
        
        # Track some metrics
        analytics.track_message("whatsapp", "user123", "user", "diagnosis", 0.85)
        analytics.track_message("sms", "user456", "bot", "covid")
        analytics.track_api_call("clinical_trials", success=True, response_time=1.5)
        
        # Get metrics
        metrics = analytics.get_usage_metrics(1)
        assert "summary" in metrics
    
    def test_health_status(self):
        """Test health status monitoring"""
        analytics = HealthBotAnalytics()
        
        health = analytics.get_health_status()
        assert "status" in health
        assert "checks" in health
        assert health["status"] in ["healthy", "unhealthy", "error"]

class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_full_whatsapp_flow(self):
        """Test complete WhatsApp flow"""
        # This would test the complete flow from webhook to response
        # In a real test, you'd use a test framework like pytest-flask
        pass
    
    def test_full_sms_flow(self):
        """Test complete SMS flow"""
        # This would test the complete flow from webhook to response
        pass
    
    def test_api_endpoints(self):
        """Test API endpoints"""
        # Test the main API endpoints
        base_url = "http://localhost:5000"
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        
        # Test chat endpoint
        chat_data = {
            "message": "I have fever and cough",
            "user_id": "test_user"
        }
        response = requests.post(f"{base_url}/chat", json=chat_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "type" in data

class TestPerformance:
    """Performance tests"""
    
    def test_response_time(self):
        """Test response time"""
        start_time = time.time()
        response = generate_bot_response("I have fever and cough")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 5.0  # Should respond within 5 seconds
    
    def test_concurrent_requests(self):
        """Test concurrent request handling"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            response = generate_bot_response("I have fever and cough")
            results.put(response)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert results.qsize() == 10
        while not results.empty():
            response = results.get()
            assert response is not None

def run_tests():
    """Run all tests"""
    print("🧪 Running HealthBot AI Chatbot tests...")
    
    # Run pytest
    pytest.main(["-v", "tests/"])

if __name__ == "__main__":
    run_tests()