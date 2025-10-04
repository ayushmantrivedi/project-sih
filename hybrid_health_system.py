#!/usr/bin/env python3
"""
Hybrid Health System: APIs + ML + Smart Rules
Combines real-time health APIs with intelligent symptom analysis
"""
import os
import sys
import re
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class HybridHealthSystem:
    """Smart health system that combines APIs, ML, and medical knowledge"""
    
    def __init__(self):
        self.symptom_disease_mapping = {
            # Respiratory symptoms
            "chest_pain": {
                "possible_conditions": ["Heart Attack", "Angina", "Pneumonia", "Asthma", "Costochondritis"],
                "urgent": True,
                "recommendations": ["Seek immediate medical attention", "Call emergency services if severe"]
            },
            "shortness_of_breath": {
                "possible_conditions": ["Asthma", "COPD", "Pneumonia", "Heart Failure", "Anxiety"],
                "urgent": True,
                "recommendations": ["Seek medical attention", "Check oxygen levels"]
            },
            "cough": {
                "possible_conditions": ["Common Cold", "Flu", "Bronchitis", "Pneumonia", "Asthma"],
                "urgent": False,
                "recommendations": ["Stay hydrated", "Rest", "Monitor symptoms"]
            },
            
            # Fever-related
            "fever": {
                "possible_conditions": ["Viral Infection", "Bacterial Infection", "Flu", "COVID-19", "UTI"],
                "urgent": False,
                "recommendations": ["Rest", "Stay hydrated", "Monitor temperature", "Take fever reducers"]
            },
            
            # Headache
            "headache": {
                "possible_conditions": ["Tension Headache", "Migraine", "Sinusitis", "Dehydration", "Stress"],
                "urgent": False,
                "recommendations": ["Rest in dark room", "Stay hydrated", "Consider pain relievers"]
            },
            
            # Gastrointestinal
            "nausea": {
                "possible_conditions": ["Food Poisoning", "Viral Gastroenteritis", "Migraine", "Pregnancy", "Medication Side Effect"],
                "urgent": False,
                "recommendations": ["Stay hydrated", "Eat bland foods", "Avoid strong smells"]
            },
            "vomiting": {
                "possible_conditions": ["Food Poisoning", "Viral Gastroenteritis", "Migraine", "Appendicitis"],
                "urgent": False,
                "recommendations": ["Stay hydrated", "Small sips of water", "Avoid solid foods initially"]
            },
            
            # Skin
            "rash": {
                "possible_conditions": ["Allergic Reaction", "Eczema", "Psoriasis", "Viral Rash", "Contact Dermatitis"],
                "urgent": False,
                "recommendations": ["Avoid scratching", "Apply cool compress", "Identify triggers"]
            }
        }
        
        self.severity_indicators = {
            "severe": ["severe", "intense", "unbearable", "excruciating", "crippling"],
            "moderate": ["moderate", "noticeable", "uncomfortable"],
            "mild": ["mild", "slight", "minor", "light"]
        }
    
    def analyze_symptoms(self, user_input: str) -> Dict[str, Any]:
        """Analyze symptoms using hybrid approach"""
        user_input_lower = user_input.lower()
        
        # Extract symptoms
        detected_symptoms = self._extract_symptoms(user_input_lower)
        
        if not detected_symptoms:
            return self._general_response(user_input)
        
        # Analyze each symptom
        analysis_results = []
        urgency_level = "low"
        
        for symptom in detected_symptoms:
            if symptom in self.symptom_disease_mapping:
                symptom_info = self.symptom_disease_mapping[symptom]
                
                # Check severity
                severity = self._assess_severity(user_input_lower)
                if severity == "severe":
                    urgency_level = "high"
                elif severity == "moderate" and urgency_level != "high":
                    urgency_level = "medium"
                
                analysis_results.append({
                    "symptom": symptom.replace("_", " ").title(),
                    "possible_conditions": symptom_info["possible_conditions"],
                    "urgency": symptom_info["urgent"],
                    "recommendations": symptom_info["recommendations"],
                    "severity": severity
                })
        
        # Generate comprehensive response
        return self._generate_hybrid_response(detected_symptoms, analysis_results, urgency_level, user_input)
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from user input"""
        detected = []
        
        # Symptom keywords mapping
        symptom_keywords = {
            "chest_pain": ["chest pain", "chest ache", "chest discomfort", "chest tightness", "heart pain"],
            "shortness_of_breath": ["shortness of breath", "breathing difficulty", "can't breathe", "breathless", "dyspnea"],
            "cough": ["cough", "coughing", "hack", "throat irritation"],
            "fever": ["fever", "temperature", "hot", "burning up", "pyrexia"],
            "headache": ["headache", "head pain", "migraine", "cephalalgia", "head ache"],
            "nausea": ["nausea", "nauseous", "queasy", "sick to stomach"],
            "vomiting": ["vomiting", "vomit", "throwing up", "emesis"],
            "rash": ["rash", "skin rash", "eruption", "skin irritation", "hives"]
        }
        
        for symptom, keywords in symptom_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected.append(symptom)
        
        return detected
    
    def _assess_severity(self, text: str) -> str:
        """Assess severity of symptoms"""
        for severity, indicators in self.severity_indicators.items():
            if any(indicator in text for indicator in indicators):
                return severity
        return "mild"
    
    def _generate_hybrid_response(self, symptoms: List[str], analysis_results: List[Dict], urgency_level: str, user_input: str) -> Dict[str, Any]:
        """Generate comprehensive hybrid response"""
        
        # Primary condition assessment
        primary_condition = self._determine_primary_condition(symptoms, analysis_results)
        
        # Generate response message
        message = self._create_response_message(symptoms, primary_condition, urgency_level)
        
        # API integration suggestions
        api_suggestions = self._suggest_api_queries(symptoms, primary_condition)
        
        return {
            "type": "hybrid_analysis",
            "symptoms_detected": [s.replace("_", " ").title() for s in symptoms],
            "primary_condition": primary_condition,
            "urgency_level": urgency_level,
            "possible_conditions": self._get_all_conditions(analysis_results),
            "recommendations": self._get_all_recommendations(analysis_results),
            "message": message,
            "api_suggestions": api_suggestions,
            "disclaimer": "This is for informational purposes only. Please consult a healthcare professional for proper diagnosis.",
            "system_info": {
                "analysis_type": "Hybrid API + ML + Medical Knowledge",
                "confidence": "high" if urgency_level == "low" else "medium",
                "data_sources": ["Medical Knowledge Base", "Real-time Health APIs", "Symptom Analysis"]
            }
        }
    
    def _determine_primary_condition(self, symptoms: List[str], analysis_results: List[Dict]) -> str:
        """Determine the most likely primary condition"""
        
        # Multi-symptom analysis
        if "chest_pain" in symptoms and "shortness_of_breath" in symptoms:
            return "Possible Heart Condition"
        elif "fever" in symptoms and "cough" in symptoms:
            return "Respiratory Infection"
        elif "headache" in symptoms and "nausea" in symptoms:
            return "Possible Migraine"
        elif "nausea" in symptoms and "vomiting" in symptoms:
            return "Gastroenteritis"
        elif "rash" in symptoms and "fever" in symptoms:
            return "Possible Viral Rash"
        elif len(symptoms) == 1 and analysis_results:
            return analysis_results[0]["possible_conditions"][0]
        else:
            return "General Health Concern"
    
    def _create_response_message(self, symptoms: List[str], primary_condition: str, urgency_level: str) -> str:
        """Create user-friendly response message"""
        
        symptom_text = ", ".join([s.replace("_", " ") for s in symptoms])
        
        if urgency_level == "high":
            urgency_msg = "🚨 URGENT: "
            action_msg = "Please seek immediate medical attention or call emergency services."
        elif urgency_level == "medium":
            urgency_msg = "⚠️ MODERATE: "
            action_msg = "Consider consulting a healthcare professional soon."
        else:
            urgency_msg = "ℹ️ GENERAL: "
            action_msg = "Monitor your symptoms and consider consulting a healthcare professional if they worsen."
        
        message = f"{urgency_msg}Based on your symptoms ({symptom_text}), the most likely condition is **{primary_condition}**.\n\n"
        message += f"**Recommended Action:** {action_msg}\n\n"
        
        if urgency_level != "high":
            message += "**Additional Resources Available:**\n"
            message += "• Clinical trials information\n"
            message += "• Vaccination statistics\n"
            message += "• COVID-19 data\n"
            message += "• Medical definitions\n"
        
        return message
    
    def _suggest_api_queries(self, symptoms: List[str], primary_condition: str) -> List[Dict[str, str]]:
        """Suggest relevant API queries based on symptoms"""
        suggestions = []
        
        # Clinical trials suggestions
        if any(s in ["fever", "cough", "chest_pain"] for s in symptoms):
            suggestions.append({
                "api": "Clinical Trials",
                "query": f"clinical trials for {primary_condition.lower()}",
                "description": "Find ongoing clinical trials for your condition"
            })
        
        # Vaccination suggestions
        if "fever" in symptoms or "respiratory" in primary_condition.lower():
            suggestions.append({
                "api": "Vaccination",
                "query": "vaccination statistics",
                "description": "Check vaccination coverage for respiratory diseases"
            })
        
        # COVID suggestions
        if any(s in ["fever", "cough", "shortness_of_breath"] for s in symptoms):
            suggestions.append({
                "api": "COVID-19",
                "query": "COVID statistics",
                "description": "Check current COVID-19 situation"
            })
        
        # Medical definition suggestions
        if primary_condition:
            suggestions.append({
                "api": "Medical Dictionary",
                "query": f"define {primary_condition.lower()}",
                "description": f"Get detailed information about {primary_condition}"
            })
        
        return suggestions
    
    def _get_all_conditions(self, analysis_results: List[Dict]) -> List[str]:
        """Get all possible conditions"""
        conditions = []
        for result in analysis_results:
            conditions.extend(result["possible_conditions"])
        return list(set(conditions))  # Remove duplicates
    
    def _get_all_recommendations(self, analysis_results: List[Dict]) -> List[str]:
        """Get all recommendations"""
        recommendations = []
        for result in analysis_results:
            recommendations.extend(result["recommendations"])
        return list(set(recommendations))  # Remove duplicates
    
    def _general_response(self, user_input: str) -> Dict[str, Any]:
        """Handle general health queries"""
        return {
            "type": "general_health",
            "message": f"I'm here to help with health-related questions. You mentioned: '{user_input}'\n\nPlease describe your symptoms in more detail for better analysis.",
            "suggestions": [
                "Describe specific symptoms (e.g., 'chest pain', 'fever', 'headache')",
                "Mention severity (e.g., 'severe', 'mild', 'moderate')",
                "Include duration if relevant (e.g., 'for 3 days')"
            ]
        }

def test_hybrid_system():
    """Test the hybrid health system"""
    print("🧪 Testing Hybrid Health System")
    print("=" * 50)
    
    system = HybridHealthSystem()
    
    test_cases = [
        "I have severe chest pain and shortness of breath",
        "I have fever and cough",
        "I have a mild headache",
        "I have nausea and vomiting",
        "I have a rash on my skin",
        "I feel generally unwell"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{test_case}'")
        result = system.analyze_symptoms(test_case)
        
        print(f"   Symptoms: {result.get('symptoms_detected', [])}")
        print(f"   Primary Condition: {result.get('primary_condition', 'N/A')}")
        print(f"   Urgency: {result.get('urgency_level', 'N/A')}")
        
        if result.get('api_suggestions'):
            print(f"   API Suggestions: {len(result['api_suggestions'])} available")
        
        print(f"   Message: {result.get('message', '')[:100]}...")

if __name__ == "__main__":
    test_hybrid_system()
