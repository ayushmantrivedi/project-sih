from mcp.server.fastmcp import FastMCP
from utils import get_search_engine, get_vector_db_manager, get_logger
import os

logger = get_logger("mcp_server")

# Initialize the FastMCP server
# We'll name it "MediAgent-MCP" to focus on medical intelligence
mcp = FastMCP("MediAgent-MCP")

@mcp.tool()
def search_and_rank_medical_data(query: str) -> str:
    """
    Searches the web for medical data and returns the top ranked results.
    Use this for fetching the latest guidelines, clinical trials, or disease info.
    """
    logger.info(f"Tool called: search_and_rank_medical_data with query: {query}")
    try:
        engine = get_search_engine()
        results = engine.get_best_results(query)
        
        if not results:
            return "No relevant results found on the web."
            
        formatted_results = []
        for i, res in enumerate(results):
            formatted_results.append(f"Source [{i+1}]: {res['text']}\nURL: {res['meta']['url']}\nRelevancy Score: {res['score']:.2f}")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error in search_and_rank tool: {str(e)}")
        return f"Error fetching data: {str(e)}"

@mcp.tool()
def get_user_clinical_history(user_id: str, query: str) -> str:
    """
    Retrieves relevant past interactions and clinical history for a specific user.
    Use this to maintain longitudinal context across chats.
    """
    logger.info(f"Tool called: get_user_clinical_history for user: {user_id}")
    try:
        db = get_vector_db_manager()
        history = db.query_history(user_id, query)
        
        if not history:
            return "No matching previous history found for this user."
            
        return "\n---\n".join(history)
    except Exception as e:
        logger.error(f"Error in get_user_clinical_history tool: {str(e)}")
        return f"Error retrieving history: {str(e)}"

@mcp.tool()
def emergency_user_alert(severity_score: int, reasoning: str) -> str:
    """
    Triggers a high-priority safety alert for the user if a condition is serious.
    severity_score: 1 (Minor) to 10 (Critical)
    reasoning: Clinical justification for the alert.
    """
    # In a real app, this might trigger a special UI element, a loud sound, 
    # or a persistent notification. For now, we return the alert message.
    alert_msg = f"""
    ⚠️ IMMEDIATE ATTENTION REQUIRED (Severity: {severity_score}/10)
    
    My analysis suggests a serious medical concern:
    {reasoning}
    
    ACTION RECOMMENDED: Please consult a healthcare professional immediately.
    If you are experiencing severe distress, call emergency services (e.g., 911 or 108).
    """
    return alert_msg

if __name__ == "__main__":
    # Run the server using stdio transport for local integration
    # For a real web app, we might use SSE transport.
    mcp.run(transport="stdio")
