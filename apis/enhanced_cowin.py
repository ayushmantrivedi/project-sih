"""
Enhanced CoWIN API integration with error handling and caching
"""
import requests
import time
from typing import Dict, List, Optional, Any
from utils import get_logger, log_api_call
from config import get_config

logger = get_logger("cowin_api")

class CoWINAPI:
    """Enhanced CoWIN API client"""
    
    def __init__(self):
        self.config = get_config()
        self.base_url = self.config.api.COWIN_BASE_URL
        self.timeout = 30
        self.retry_attempts = 3
        
        # API endpoints
        self.endpoints = {
            "vaccination_by_age": "/vaccine/vaccination-by-age",
            "vaccination_by_state": "/vaccine/vaccination-by-state",
            "session_calendar": "/appointment/sessions/calendarByDistrict",
            "session_calendar_by_pin": "/appointment/sessions/calendarByPin"
        }
    
    def get_vaccination_stats(self, state_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Get vaccination statistics
        
        Args:
            state_code: Optional state code for state-specific data
        
        Returns:
            Dict containing vaccination statistics
        """
        start_time = time.time()
        
        try:
            # Choose endpoint based on parameters
            if state_code:
                endpoint = self.endpoints["vaccination_by_state"]
                url = f"{self.base_url}{endpoint}"
                params = {"state_code": state_code}
            else:
                endpoint = self.endpoints["vaccination_by_age"]
                url = f"{self.base_url}{endpoint}"
                params = {}
            
            logger.info(f"Fetching CoWIN vaccination stats: {endpoint}")
            
            for attempt in range(self.retry_attempts):
                try:
                    response = requests.get(
                        url,
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
                        log_api_call("CoWIN", url, response.status_code, response_time)
                        
                        # Parse and format results
                        stats = self._parse_vaccination_stats(response.json(), state_code)
                        
                        return {
                            "success": True,
                            "data": stats,
                            "response_time": response_time,
                            "api_source": "CoWIN API",
                            "endpoint": endpoint
                        }
                    
                    else:
                        logger.warning(f"CoWIN API returned status {response.status_code}")
                        if attempt < self.retry_attempts - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        
                        # Log failed API call
                        log_api_call("CoWIN", url, response.status_code, response_time, 
                                   f"HTTP {response.status_code}")
                        
                        return {
                            "success": False,
                            "error": f"API returned status {response.status_code}",
                            "response_time": response_time
                        }
                
                except requests.exceptions.Timeout:
                    logger.warning(f"CoWIN API timeout (attempt {attempt + 1})")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(2 ** attempt)
                        continue
                    
                    log_api_call("CoWIN", url, 408, response_time, "Timeout")
                    return {
                        "success": False,
                        "error": "API request timeout",
                        "response_time": response_time
                    }
                
                except requests.exceptions.RequestException as e:
                    logger.warning(f"CoWIN API request error (attempt {attempt + 1}): {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(2 ** attempt)
                        continue
                    
                    log_api_call("CoWIN", url, 500, response_time, str(e))
                    return {
                        "success": False,
                        "error": f"Request error: {str(e)}",
                        "response_time": response_time
                    }
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Unexpected error in CoWIN API: {e}")
            log_api_call("CoWIN", self.base_url, 500, response_time, str(e))
            
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "response_time": response_time
            }
    
    def get_session_calendar(self, district_id: str, date: str) -> Dict[str, Any]:
        """
        Get vaccination session calendar by district
        
        Args:
            district_id: District ID
            date: Date in DD-MM-YYYY format
        
        Returns:
            Dict containing session calendar data
        """
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{self.endpoints['session_calendar']}"
            params = {
                "district_id": district_id,
                "date": date
            }
            
            logger.info(f"Fetching CoWIN session calendar for district {district_id} on {date}")
            
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout,
                headers={
                    "User-Agent": "HealthBot-AI-Chatbot/1.0",
                    "Accept": "application/json"
                }
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                log_api_call("CoWIN", url, response.status_code, response_time)
                
                sessions = self._parse_session_calendar(response.json())
                
                return {
                    "success": True,
                    "data": sessions,
                    "response_time": response_time,
                    "api_source": "CoWIN API"
                }
            else:
                log_api_call("CoWIN", url, response.status_code, response_time, 
                           f"HTTP {response.status_code}")
                return {
                    "success": False,
                    "error": f"API returned status {response.status_code}",
                    "response_time": response_time
                }
        
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Error in CoWIN session calendar API: {e}")
            log_api_call("CoWIN", self.base_url, 500, response_time, str(e))
            
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "response_time": response_time
            }
    
    def _parse_vaccination_stats(self, json_data: Any, state_code: Optional[str] = None) -> Dict[str, Any]:
        """Parse and format vaccination statistics"""
        try:
            if isinstance(json_data, dict):
                # Handle different response structures
                if "vaccinationByAge" in json_data:
                    return self._format_age_wise_stats(json_data["vaccinationByAge"])
                elif "topBlock" in json_data:
                    return self._format_state_wise_stats(json_data)
                else:
                    return self._format_general_stats(json_data)
            elif isinstance(json_data, list):
                return {"vaccination_data": json_data, "type": "list"}
            else:
                return {"raw_data": str(json_data), "type": "unknown"}
                
        except Exception as e:
            logger.error(f"Error parsing CoWIN vaccination stats: {e}")
            return {"error": "Failed to parse vaccination statistics"}
    
    def _format_age_wise_stats(self, age_data: List[Dict]) -> Dict[str, Any]:
        """Format age-wise vaccination statistics"""
        try:
            formatted_stats = {
                "type": "age_wise_vaccination",
                "total_doses": 0,
                "age_groups": []
            }
            
            for age_group in age_data:
                if isinstance(age_group, dict):
                    formatted_group = {
                        "age_group": age_group.get("age", "Unknown"),
                        "doses_1": age_group.get("dose1", 0),
                        "doses_2": age_group.get("dose2", 0),
                        "total": age_group.get("total", 0)
                    }
                    formatted_stats["age_groups"].append(formatted_group)
                    formatted_stats["total_doses"] += formatted_group["total"]
            
            return formatted_stats
            
        except Exception as e:
            logger.error(f"Error formatting age-wise stats: {e}")
            return {"error": "Failed to format age-wise statistics"}
    
    def _format_state_wise_stats(self, state_data: Dict) -> Dict[str, Any]:
        """Format state-wise vaccination statistics"""
        try:
            top_block = state_data.get("topBlock", {})
            
            return {
                "type": "state_wise_vaccination",
                "total_vaccinations": top_block.get("vaccination", {}).get("total", 0),
                "total_doses_1": top_block.get("vaccination", {}).get("tot_dose_1", 0),
                "total_doses_2": top_block.get("vaccination", {}).get("tot_dose_2", 0),
                "last_updated": top_block.get("lastUpdated", ""),
                "source": "CoWIN API"
            }
            
        except Exception as e:
            logger.error(f"Error formatting state-wise stats: {e}")
            return {"error": "Failed to format state-wise statistics"}
    
    def _format_general_stats(self, data: Dict) -> Dict[str, Any]:
        """Format general vaccination statistics"""
        return {
            "type": "general_vaccination",
            "data": data,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _parse_session_calendar(self, json_data: Any) -> Dict[str, Any]:
        """Parse session calendar data"""
        try:
            if isinstance(json_data, dict) and "centers" in json_data:
                centers = json_data["centers"]
                return {
                    "total_centers": len(centers),
                    "centers": centers,
                    "date": json_data.get("date", ""),
                    "type": "session_calendar"
                }
            else:
                return {
                    "type": "session_calendar",
                    "raw_data": json_data
                }
                
        except Exception as e:
            logger.error(f"Error parsing session calendar: {e}")
            return {"error": "Failed to parse session calendar"}

# Global API instance
_cowin_api = None

def get_cowin_api() -> CoWINAPI:
    """Get the global CoWIN API instance"""
    global _cowin_api
    if _cowin_api is None:
        _cowin_api = CoWINAPI()
    return _cowin_api

def get_cowin_stats(state_code: Optional[str] = None) -> Dict[str, Any]:
    """Get CoWIN vaccination statistics - enhanced version"""
    api = get_cowin_api()
    return api.get_vaccination_stats(state_code)

# Backward compatibility
def get_cowin_stats_legacy() -> Dict:
    """Backward compatible function"""
    result = get_cowin_stats()
    if result.get("success"):
        return result.get("data", {})
    return {}
