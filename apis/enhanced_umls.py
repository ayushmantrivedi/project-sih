"""
Enhanced UMLS API integration with error handling and fallback
"""
import requests
import time
from typing import Dict, List, Optional, Any
from utils import get_logger, log_api_call
from config import get_config

logger = get_logger("umls_api")

class UMLSAPI:
    """Enhanced UMLS API client"""
    
    def __init__(self):
        self.config = get_config()
        self.base_url = self.config.api.UMLS_BASE_URL
        self.api_key = self.config.api.UMLS_API_KEY
        self.timeout = 30
        self.retry_attempts = 3
        
        # Fallback medical dictionary for demo purposes
        self.fallback_dictionary = {
            "hypertension": {
                "definition": "High blood pressure, a long-term medical condition in which the blood pressure in the arteries is persistently elevated.",
                "synonyms": ["high blood pressure", "arterial hypertension"],
                "cui": "C0020538",
                "source": "fallback"
            },
            "diabetes": {
                "definition": "A group of metabolic disorders characterized by high blood sugar levels over a prolonged period.",
                "synonyms": ["diabetes mellitus", "high blood sugar"],
                "cui": "C0011860",
                "source": "fallback"
            },
            "fever": {
                "definition": "An abnormally high body temperature, usually as a result of illness.",
                "synonyms": ["pyrexia", "hyperthermia"],
                "cui": "C0015967",
                "source": "fallback"
            },
            "cough": {
                "definition": "A reflex action to clear the throat and breathing passage of irritants.",
                "synonyms": ["hacking", "throat clearing"],
                "cui": "C0010200",
                "source": "fallback"
            },
            "headache": {
                "definition": "Pain in the head or upper neck area.",
                "synonyms": ["cephalalgia", "head pain"],
                "cui": "C0018681",
                "source": "fallback"
            },
            "nausea": {
                "definition": "A feeling of sickness with an inclination to vomit.",
                "synonyms": ["queasiness", "sickness"],
                "cui": "C0027497",
                "source": "fallback"
            },
            "vomiting": {
                "definition": "The forcible voluntary or involuntary emptying of stomach contents through the mouth.",
                "synonyms": ["emesis", "throwing up"],
                "cui": "C0042963",
                "source": "fallback"
            },
            "rash": {
                "definition": "A change in the skin which affects its color, appearance, or texture.",
                "synonyms": ["skin eruption", "dermatitis"],
                "cui": "C0033774",
                "source": "fallback"
            }
        }
    
    def get_medical_definition(self, term: str, search_type: str = "exact") -> Dict[str, Any]:
        """
        Get medical definition from UMLS
        
        Args:
            term: Medical term to look up
            search_type: Type of search (exact, contains, starts)
        
        Returns:
            Dict containing medical definition and related information
        """
        start_time = time.time()
        
        # Check if we have a valid API key
        if not self.api_key or self.api_key == "your_umls_api_key_here":
            logger.info(f"No valid UMLS API key, using fallback for term: {term}")
            return self._get_fallback_definition(term, start_time)
        
        try:
            # Build search URL
            search_url = f"{self.base_url}/search/version"
            params = {
                "apiKey": self.api_key,
                "string": term,
                "searchType": search_type,
                "pageSize": 10,
                "pageNumber": 1
            }
            
            logger.info(f"Searching UMLS for term: {term}")
            
            for attempt in range(self.retry_attempts):
                try:
                    response = requests.get(
                        search_url,
                        params=params,
                        timeout=self.timeout,
                        headers={
                            "User-Agent": "HealthBot-AI-Chatbot/1.0",
                            "Accept": "application/json"
                        }
                    )
                    
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        # Log successful API call
                        log_api_call("UMLS", search_url, response.status_code, response_time)
                        
                        # Parse and format results
                        definition = self._parse_umls_results(response.json(), term)
                        
                        return {
                            "success": True,
                            "term": term,
                            "definition": definition,
                            "response_time": response_time,
                            "api_source": "UMLS API"
                        }
                    
                    elif response.status_code == 401:
                        logger.warning("UMLS API authentication failed")
                        log_api_call("UMLS", search_url, response.status_code, response_time, "Authentication failed")
                        return self._get_fallback_definition(term, start_time)
                    
                    else:
                        logger.warning(f"UMLS API returned status {response.status_code}")
                        if attempt < self.retry_attempts - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        
                        # Use fallback if API fails
                        log_api_call("UMLS", search_url, response.status_code, response_time, 
                                   f"HTTP {response.status_code}")
                        return self._get_fallback_definition(term, start_time)
                
                except requests.exceptions.Timeout:
                    logger.warning(f"UMLS API timeout (attempt {attempt + 1})")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(2 ** attempt)
                        continue
                    
                    return self._get_fallback_definition(term, start_time)
                
                except requests.exceptions.RequestException as e:
                    logger.warning(f"UMLS API request error (attempt {attempt + 1}): {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(2 ** attempt)
                        continue
                    
                    return self._get_fallback_definition(term, start_time)
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Unexpected error in UMLS API: {e}")
            log_api_call("UMLS", self.base_url, 500, response_time, str(e))
            return self._get_fallback_definition(term, start_time)
    
    def _get_fallback_definition(self, term: str, start_time: float) -> Dict[str, Any]:
        """Get definition from fallback dictionary"""
        response_time = time.time() - start_time
        
        # Normalize term for lookup
        normalized_term = term.lower().strip()
        
        # Direct lookup
        if normalized_term in self.fallback_dictionary:
            definition = self.fallback_dictionary[normalized_term].copy()
            definition["api_source"] = "Fallback Dictionary"
            definition["response_time"] = response_time
            
            return {
                "success": True,
                "term": term,
                "definition": definition,
                "response_time": response_time,
                "api_source": "Fallback Dictionary"
            }
        
        # Fuzzy search for partial matches
        for key, value in self.fallback_dictionary.items():
            if normalized_term in key or any(normalized_term in syn.lower() for syn in value.get("synonyms", [])):
                definition = value.copy()
                definition["api_source"] = "Fallback Dictionary (Fuzzy Match)"
                definition["response_time"] = response_time
                
                return {
                    "success": True,
                    "term": term,
                    "definition": definition,
                    "response_time": response_time,
                    "api_source": "Fallback Dictionary (Fuzzy Match)",
                    "matched_term": key
                }
        
        # No match found
        return {
            "success": False,
            "term": term,
            "error": "Term not found in medical dictionary",
            "suggestion": "Please check spelling or try a different medical term",
            "available_terms": list(self.fallback_dictionary.keys()),
            "response_time": response_time,
            "api_source": "Fallback Dictionary"
        }
    
    def _parse_umls_results(self, json_data: Any, term: str) -> Dict[str, Any]:
        """Parse UMLS API results"""
        try:
            if isinstance(json_data, dict) and "result" in json_data:
                results = json_data["result"]
                if results.get("results"):
                    # Get the first result (most relevant)
                    first_result = results["results"][0]
                    
                    return {
                        "definition": first_result.get("definition", "No definition available"),
                        "cui": first_result.get("ui", ""),
                        "concept_name": first_result.get("name", term),
                        "semantic_types": first_result.get("semanticTypes", []),
                        "atoms": first_result.get("atoms", []),
                        "source": "UMLS API",
                        "relevance_score": first_result.get("score", 0)
                    }
                else:
                    return {
                        "error": "No results found",
                        "term": term,
                        "source": "UMLS API"
                    }
            else:
                return {
                    "error": "Invalid response format",
                    "raw_data": str(json_data)[:500],  # Truncate for logging
                    "source": "UMLS API"
                }
                
        except Exception as e:
            logger.error(f"Error parsing UMLS results: {e}")
            return {
                "error": f"Failed to parse UMLS response: {str(e)}",
                "source": "UMLS API"
            }
    
    def search_similar_terms(self, term: str) -> Dict[str, Any]:
        """Search for similar medical terms"""
        similar_terms = []
        
        # Simple similarity search in fallback dictionary
        normalized_term = term.lower().strip()
        
        for key, value in self.fallback_dictionary.items():
            # Check if terms are similar (simple string matching)
            if (normalized_term in key or 
                any(normalized_term in syn.lower() for syn in value.get("synonyms", [])) or
                key.startswith(normalized_term[:3])):
                similar_terms.append({
                    "term": key,
                    "definition": value.get("definition", ""),
                    "similarity": "partial_match"
                })
        
        return {
            "success": len(similar_terms) > 0,
            "term": term,
            "similar_terms": similar_terms[:5],  # Limit to 5 results
            "total_found": len(similar_terms)
        }

# Global API instance
_umls_api = None

def get_umls_api() -> UMLSAPI:
    """Get the global UMLS API instance"""
    global _umls_api
    if _umls_api is None:
        _umls_api = UMLSAPI()
    return _umls_api

def get_umls_info(term: str, search_type: str = "exact") -> Dict[str, Any]:
    """Get UMLS medical definition - enhanced version"""
    api = get_umls_api()
    return api.get_medical_definition(term, search_type)

# Backward compatibility
def get_umls_info_legacy(term: str) -> Dict:
    """Backward compatible function"""
    result = get_umls_info(term)
    if result.get("success"):
        return {
            "term": result.get("term", term),
            "definition": result.get("definition", {}).get("definition", "No definition available"),
            "source": result.get("api_source", "Unknown")
        }
    return {
        "term": term,
        "definition": "UMLS API requires license. Demo only.",
        "source": "Fallback"
    }
