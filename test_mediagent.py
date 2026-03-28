import sys
import os
from chatbot import generate_bot_response
import json

def test_mediagent_flow():
    print("🚀 Testing MediAgent A2A/MCP Flow...")
    
    # Mock user message
    user_message = "I have a sudden sharp pain in my left arm and I feel dizzy. Is this an emergency?"
    user_id = "test_user_001"
    
    print(f"📝 User Query: {user_message}")
    print(f"🆔 User ID: {user_id}")
    print("-" * 50)
    
    try:
        # Run the orchestrator
        response = generate_bot_response(user_message, user_id=user_id)
        
        print("\n✅ Response Received:")
        print(json.dumps(response, indent=2))
        
        if response.get("type") == "agentic_reasoning":
            print("\n🌟 Analysis Result:")
            print(response["content"])
            print(f"\n📊 Evidence Used: {response['metadata']['evidence_used']} sources")
        
    except Exception as e:
        print(f"❌ Flow failed: {str(e)}")

if __name__ == "__main__":
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    test_mediagent_flow()
