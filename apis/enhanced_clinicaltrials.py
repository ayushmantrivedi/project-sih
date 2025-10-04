"""
Enhanced ClinicalTrials.gov API integration with error handling and caching
"""
import requests
import time
from typing import List, Dict, Optional, Any
from utils import get_logger, log_api_call
from config import get_config

logger = get_logger("clinicaltrials_api")

class ClinicalTrialsAPI:
    """Enhanced ClinicalTrials.gov API client"""
    
    def __init__(self):
        self.config = get_config()
        self.base_url = self.config.api.CLINICAL_TRIALS_BASE_URL
        self.max_results = self.config.api.CLINICAL_TRIALS_MAX_RESULTS
        self.timeout = 30
        self.retry_attempts = 3
        
    def search_trials(self, query: str, condition: Optional[str] = None, 
                     location: Optional[str] = None, phase: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for clinical trials with enhanced parameters
        
        Args:
            query: Search query (disease, treatment, etc.)
            condition: Specific medical condition
            location: Geographic location
            phase: Trial phase (Phase I, Phase II, etc.)
        
        Returns:
            Dict containing trial results and metadata
        """
        start_time = time.time()
        
        try:
            # Build search expression
            search_expr = self._build_search_expression(query, condition, location, phase)
            
            # Make API request
            params = {
                "expr": search_expr,
                "min_rnk": 1,
                "max_rnk": self.max_results,
                "fmt": "json"
            }
            
            logger.info(f"Searching clinical trials: {search_expr}")
            
            for attempt in range(self.retry_attempts):
                try:
                    response = requests.get(
                        self.base_url,
                        params=params,
                        timeout=self.timeout
                    )
                    
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        # Log successful API call
                        log_api_call("ClinicalTrials", self.base_url, response.status_code, response_time)
                        
                        # Parse and format results
                        results = self._parse_trial_results(response.json())
                        
                        return {
                            "success": True,
                            "query": search_expr,
                            "total_results": results.get("total", 0),
                            "trials": results.get("trials", []),
                            "response_time": response_time,
                            "api_source": "ClinicalTrials.gov"
                        }
                    
                    else:
                        logger.warning(f"ClinicalTrials API returned status {response.status_code}")
                        if attempt < self.retry_attempts - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        
                        # Log failed API call
                        log_api_call("ClinicalTrials", self.base_url, response.status_code, response_time, 
                                   f"HTTP {response.status_code}")
                        
                        return {
                            "success": False,
                            "error": f"API returned status {response.status_code}",
                            "query": search_expr,
                            "response_time": response_time
                        }
                
                except requests.exceptions.Timeout:
                    logger.warning(f"ClinicalTrials API timeout (attempt {attempt + 1})")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(2 ** attempt)
                        continue
                    
                    log_api_call("ClinicalTrials", self.base_url, 408, response_time, "Timeout")
                    return {
                        "success": False,
                        "error": "API request timeout",
                        "query": search_expr,
                        "response_time": response_time
                    }
                
                except requests.exceptions.RequestException as e:
                    logger.warning(f"ClinicalTrials API request error (attempt {attempt + 1}): {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(2 ** attempt)
                        continue
                    
                    log_api_call("ClinicalTrials", self.base_url, 500, response_time, str(e))
                    return {
                        "success": False,
                        "error": f"Request error: {str(e)}",
                        "query": search_expr,
                        "response_time": response_time
                    }
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Unexpected error in ClinicalTrials API: {e}")
            log_api_call("ClinicalTrials", self.base_url, 500, response_time, str(e))
            
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "query": query,
                "response_time": response_time
            }
    
    def _build_search_expression(self, query: str, condition: Optional[str] = None,
                                location: Optional[str] = None, phase: Optional[str] = None) -> str:
        """Build search expression for ClinicalTrials API"""
        expr_parts = [f'"{query}"']
        
        if condition:
            expr_parts.append(f'CONDITION="{condition}"')
        
        if location:
            expr_parts.append(f'SEARCH[Location](AREA[LocationCountry]"{location}")')
        
        if phase:
            expr_parts.append(f'PHASE="{phase}"')
        
        return " AND ".join(expr_parts)
    
    def _parse_trial_results(self, json_response: Dict) -> Dict[str, Any]:
        """Parse and format trial results from API response"""
        try:
            full_studies = json_response.get("FullStudiesResponse", {})
            studies = full_studies.get("FullStudies", [])
            
            trials = []
            for study in studies:
                try:
                    protocol = study.get("Study", {}).get("ProtocolSection", {})
                    
                    # Extract trial information
                    identification = protocol.get("IdentificationModule", {})
                    conditions = protocol.get("ConditionsModule", {})
                    eligibility = protocol.get("EligibilityModule", {})
                    design = protocol.get("DesignModule", {})
                    status = protocol.get("StatusModule", {})
                    
                    trial = {
                        "nct_id": identification.get("NCTId", ""),
                        "title": identification.get("BriefTitle", ""),
                        "official_title": identification.get("OfficialTitle", ""),
                        "summary": identification.get("BriefSummary", ""),
                        "conditions": conditions.get("ConditionList", {}).get("Condition", []),
                        "phase": design.get("PhaseList", {}).get("Phase", []),
                        "status": status.get("OverallStatus", ""),
                        "start_date": status.get("StartDateStruct", {}).get("Date", ""),
                        "completion_date": status.get("CompletionDateStruct", {}).get("Date", ""),
                        "sponsors": self._extract_sponsors(protocol.get("SponsorCollaboratorsModule", {})),
                        "locations": self._extract_locations(protocol.get("OversightModule", {})),
                        "eligibility_criteria": eligibility.get("EligibilityCriteria", ""),
                        "study_type": design.get("StudyType", ""),
                        "intervention_model": design.get("InterventionModel", ""),
                        "primary_purpose": design.get("PrimaryPurpose", "")
                    }
                    
                    trials.append(trial)
                    
                except Exception as e:
                    logger.warning(f"Error parsing individual trial: {e}")
                    continue
            
            return {
                "total": len(trials),
                "trials": trials
            }
            
        except Exception as e:
            logger.error(f"Error parsing trial results: {e}")
            return {"total": 0, "trials": []}
    
    def _extract_sponsors(self, sponsor_module: Dict) -> List[Dict]:
        """Extract sponsor information"""
        sponsors = []
        try:
            lead_sponsor = sponsor_module.get("LeadSponsor", {})
            if lead_sponsor:
                sponsors.append({
                    "name": lead_sponsor.get("Name", ""),
                    "type": "Lead Sponsor"
                })
            
            collaborators = sponsor_module.get("CollaboratorList", {}).get("Collaborator", [])
            for collab in collaborators:
                sponsors.append({
                    "name": collab.get("Name", ""),
                    "type": "Collaborator"
                })
                
        except Exception as e:
            logger.warning(f"Error extracting sponsors: {e}")
        
        return sponsors
    
    def _extract_locations(self, oversight_module: Dict) -> List[Dict]:
        """Extract location information"""
        locations = []
        try:
            # This is a simplified extraction - the actual API structure may vary
            # You would need to examine the actual API response structure
            pass
        except Exception as e:
            logger.warning(f"Error extracting locations: {e}")
        
        return locations

# Global API instance
_clinical_trials_api = None

def get_clinical_trials_api() -> ClinicalTrialsAPI:
    """Get the global ClinicalTrials API instance"""
    global _clinical_trials_api
    if _clinical_trials_api is None:
        _clinical_trials_api = ClinicalTrialsAPI()
    return _clinical_trials_api

def search_clinical_trials(query: str, **kwargs) -> Dict[str, Any]:
    """Search for clinical trials - enhanced version"""
    api = get_clinical_trials_api()
    return api.search_trials(query, **kwargs)

# Backward compatibility
def get_clinical_trials(query: str) -> List[Dict]:
    """Backward compatible function"""
    result = search_clinical_trials(query)
    if result.get("success"):
        return result.get("trials", [])
    return []
