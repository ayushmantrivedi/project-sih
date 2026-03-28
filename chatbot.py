from utils import get_logger, get_search_engine, get_vector_db_manager, get_llama_client
from agents.debate_manager import get_debate_manager
from mcp_server import mcp # We import the MCP instance to access tools
import json

logger = get_logger("chatbot")

def generate_bot_response(user_message, user_id="anonymous"):
    """
    Main Agent Orchestrator for MediAgent A2A/MCP Evolution.
    """
    logger.info(f"Orchestrating response for user: {user_id}")
    
    try:
        # Step 1: Researcher Agent - Live Web Scrape & Rank
        # We call the MCP tool directly as an Orchestrator action
        logger.info("Step 1: Researching web content...")
        search_engine = get_search_engine()
        results = search_engine.get_best_results(user_message)
        
        rag_evidence = ""
        if results:
            evidence_parts = []
            for i, res in enumerate(results):
                evidence_parts.append(f"Source [{i+1}]: {res['text']}\nURL: {res['meta']['url']}")
            rag_evidence = "\n\n".join(evidence_parts)
        else:
            rag_evidence = "No specific web evidence found."
            
        # Step 2: Context Agent - Retrieve User History
        logger.info("Step 2: Retrieving user history...")
        db_manager = get_vector_db_manager()
        user_history = db_manager.query_history(user_id, user_message)
        history_text = "\n---\n".join(user_history) if user_history else "No previous history."
        
        # Step 3: Reasoning Engine - A2A Debate Loop
        logger.info("Step 3: Running A2A Reasoning Debate...")
        debate_engine = get_debate_manager()
        debate_result = debate_engine.run_debate(user_message, rag_evidence, history_text)
        
        final_answer = debate_result["final_answer"]
        
        # Step 4: Persistent Memory - Store Interaction
        logger.info("Step 4: Persisting interaction...")
        db_manager.add_interaction(user_id, user_message, final_answer)
        
        # Step 5: Format for Delivery (WhatsApp/SMS Optimized)
        response_data = {
            "type": "agentic_reasoning",
            "content": final_answer,
            "status": "verified",
            "metadata": {
                "evidence_used": len(results) if results else 0,
                "history_found": True if user_history else False
            }
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in Orchestrator: {str(e)}")
        return {
            "type": "error",
            "content": "Our reasoning engine encountered an issue. Please try again soon.",
            "error_detail": str(e)
        }