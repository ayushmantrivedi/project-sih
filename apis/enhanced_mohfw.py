"""
Enhanced MoHFW COVID-19 API integration with error handling and caching
"""
import requests
import time
from typing import Dict, List, Optional, Any
from utils import get_logger, log_api_call
from config import get_config

logger = get_logger("mohfw_api")

class MoHFWAPI:
    """Enhanced MoHFW COVID-19 API client"""
    
    def __init__(self):
        self.config = get_config()
        self.base_url = self.config.api.MOHFW_BASE_URL
        self.timeout = 30
        self.retry_attempts = 3
        
        # Alternative endpoints (in case primary fails)
        self.alternative_urls = [
            "https://api.covid19india.org/data.json",
            "https://data.covid19india.org/v4/min/data.min.json"
        ]
    
    def get_covid_stats(self, state: Optional[str] = None) -> Dict[str, Any]:
        """
        Get COVID-19 statistics
        
        Args:
            state: Optional state name for state-specific data
        
        Returns:
            Dict containing COVID-19 statistics
        """
        start_time = time.time()
        
        # Try primary URL first
        for url_index, url in enumerate([self.base_url] + self.alternative_urls):
            try:
                logger.info(f"Fetching MoHFW COVID-19 data from URL {url_index + 1}")
                
                for attempt in range(self.retry_attempts):
                    try:
                        response = requests.get(
                            url,
                            timeout=self.timeout,
                            headers={
                                "User-Agent": "HealthBot-AI-Chatbot/1.0",
                                "Accept": "application/json"
                            }
                        )
                        
                        response_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            # Log successful API call
                            log_api_call("MoHFW", url, response.status_code, response_time)
                            
                            # Parse and format results
                            stats = self._parse_covid_stats(response.json(), state)
                            
                            return {
                                "success": True,
                                "data": stats,
                                "response_time": response_time,
                                "api_source": "MoHFW COVID-19 API",
                                "source_url": url
                            }
                        
                        else:
                            logger.warning(f"MoHFW API returned status {response.status_code} for URL {url_index + 1}")
                            if attempt < self.retry_attempts - 1:
                                time.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            
                            # Try next URL if this one fails
                            break
                    
                    except requests.exceptions.Timeout:
                        logger.warning(f"MoHFW API timeout for URL {url_index + 1} (attempt {attempt + 1})")
                        if attempt < self.retry_attempts - 1:
                            time.sleep(2 ** attempt)
                            continue
                        # Try next URL
                        break
                    
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"MoHFW API request error for URL {url_index + 1} (attempt {attempt + 1}): {e}")
                        if attempt < self.retry_attempts - 1:
                            time.sleep(2 ** attempt)
                            continue
                        # Try next URL
                        break
                
            except Exception as e:
                logger.error(f"Unexpected error with MoHFW URL {url_index + 1}: {e}")
                continue
        
        # If all URLs failed
        response_time = time.time() - start_time
        log_api_call("MoHFW", self.base_url, 500, response_time, "All URLs failed")
        
        return {
            "success": False,
            "error": "All COVID-19 API endpoints failed",
            "response_time": response_time
        }
    
    def get_state_wise_data(self) -> Dict[str, Any]:
        """Get state-wise COVID-19 data"""
        result = self.get_covid_stats()
        if result.get("success"):
            return {
                "success": True,
                "states": result.get("data", {}).get("statewise", []),
                "response_time": result.get("response_time", 0)
            }
        return result
    
    def get_national_summary(self) -> Dict[str, Any]:
        """Get national COVID-19 summary"""
        result = self.get_covid_stats()
        if result.get("success"):
            data = result.get("data", {})
            return {
                "success": True,
                "summary": data.get("total", {}),
                "response_time": result.get("response_time", 0)
            }
        return result
    
    def _parse_covid_stats(self, json_data: Any, state: Optional[str] = None) -> Dict[str, Any]:
        """Parse and format COVID-19 statistics"""
        try:
            if isinstance(json_data, dict):
                # Handle different response structures
                if "statewise" in json_data:
                    return self._parse_statewise_format(json_data, state)
                elif "total" in json_data:
                    return self._parse_v4_format(json_data, state)
                else:
                    return self._parse_general_format(json_data, state)
            else:
                return {"error": "Invalid response format", "type": "unknown"}
                
        except Exception as e:
            logger.error(f"Error parsing COVID-19 stats: {e}")
            return {"error": "Failed to parse COVID-19 statistics"}
    
    def _parse_statewise_format(self, data: Dict, state: Optional[str] = None) -> Dict[str, Any]:
        """Parse statewise format (covid19india.org format)"""
        try:
            statewise_data = data.get("statewise", [])
            total_data = data.get("total", {})
            
            if state:
                # Find specific state
                state_data = next((s for s in statewise_data if s.get("state", "").lower() == state.lower()), None)
                if state_data:
                    return {
                        "type": "state_data",
                        "state": state_data,
                        "state_name": state
                    }
                else:
                    return {
                        "type": "state_not_found",
                        "available_states": [s.get("state", "") for s in statewise_data],
                        "requested_state": state
                    }
            else:
                # Return all state data
                formatted_states = []
                for state_info in statewise_data:
                    if state_info.get("state") != "Total":  # Skip total row
                        formatted_states.append(self._format_state_data(state_info))
                
                return {
                    "type": "all_states",
                    "total_summary": self._format_total_data(total_data),
                    "states": formatted_states,
                    "total_states": len(formatted_states),
                    "last_updated": data.get("last_updated", "")
                }
                
        except Exception as e:
            logger.error(f"Error parsing statewise format: {e}")
            return {"error": "Failed to parse statewise data"}
    
    def _parse_v4_format(self, data: Dict, state: Optional[str] = None) -> Dict[str, Any]:
        """Parse v4 format (minimal format)"""
        try:
            total_data = data.get("total", {})
            
            if state:
                state_data = data.get(state.upper(), {})
                if state_data:
                    return {
                        "type": "state_data_v4",
                        "state": self._format_v4_state_data(state_data),
                        "state_name": state
                    }
                else:
                    return {
                        "type": "state_not_found_v4",
                        "requested_state": state
                    }
            else:
                return {
                    "type": "national_summary_v4",
                    "total": self._format_v4_total_data(total_data),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
        except Exception as e:
            logger.error(f"Error parsing v4 format: {e}")
            return {"error": "Failed to parse v4 format data"}
    
    def _parse_general_format(self, data: Dict, state: Optional[str] = None) -> Dict[str, Any]:
        """Parse general format"""
        return {
            "type": "general_format",
            "raw_data": data,
            "requested_state": state
        }
    
    def _format_state_data(self, state_info: Dict) -> Dict[str, Any]:
        """Format individual state data"""
        return {
            "state": state_info.get("state", ""),
            "confirmed": int(state_info.get("confirmed", 0)),
            "active": int(state_info.get("active", 0)),
            "recovered": int(state_info.get("recovered", 0)),
            "deaths": int(state_info.get("deaths", 0)),
            "deltaconfirmed": int(state_info.get("deltaconfirmed", 0)),
            "deltarecovered": int(state_info.get("deltarecovered", 0)),
            "deltadeaths": int(state_info.get("deltadeaths", 0)),
            "lastupdatedtime": state_info.get("lastupdatedtime", "")
        }
    
    def _format_total_data(self, total_info: Dict) -> Dict[str, Any]:
        """Format total/aggregate data"""
        return {
            "confirmed": int(total_info.get("confirmed", 0)),
            "active": int(total_info.get("active", 0)),
            "recovered": int(total_info.get("recovered", 0)),
            "deaths": int(total_info.get("deaths", 0)),
            "last_updated": total_info.get("lastupdatedtime", "")
        }
    
    def _format_v4_state_data(self, state_data: Dict) -> Dict[str, Any]:
        """Format v4 state data"""
        total = state_data.get("total", {})
        return {
            "confirmed": total.get("confirmed", 0),
            "recovered": total.get("recovered", 0),
            "deaths": total.get("deceased", 0),
            "active": total.get("confirmed", 0) - total.get("recovered", 0) - total.get("deceased", 0),
            "delta": state_data.get("delta", {}),
            "meta": state_data.get("meta", {})
        }
    
    def _format_v4_total_data(self, total_data: Dict) -> Dict[str, Any]:
        """Format v4 total data"""
        return {
            "confirmed": total_data.get("confirmed", 0),
            "recovered": total_data.get("recovered", 0),
            "deaths": total_data.get("deceased", 0),
            "active": total_data.get("confirmed", 0) - total_data.get("recovered", 0) - total_data.get("deceased", 0),
            "delta": total_data.get("delta", {})
        }

# Global API instance
_mohfw_api = None

def get_mohfw_api() -> MoHFWAPI:
    """Get the global MoHFW API instance"""
    global _mohfw_api
    if _mohfw_api is None:
        _mohfw_api = MoHFWAPI()
    return _mohfw_api

def get_mohfw_data(state: Optional[str] = None) -> Dict[str, Any]:
    """Get MoHFW COVID-19 data - enhanced version"""
    api = get_mohfw_api()
    return api.get_covid_stats(state)

# Backward compatibility
def get_mohfw_data_legacy() -> List[Dict]:
    """Backward compatible function"""
    result = get_mohfw_data()
    if result.get("success"):
        return result.get("data", {}).get("states", [])
    return []
