import chromadb
from chromadb.config import Settings
import os
import uuid
from typing import List, Dict

class VectorDBManager:
    """
    Manages per-user vector collections in ChromaDB.
    """
    def __init__(self, persist_directory="./database/vector_db"):
        self.persist_directory = persist_directory
        # Ensure path exists
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        self.client = chromadb.PersistentClient(path=persist_directory)
        
    def get_user_collection(self, user_id: str):
        """
        Retrieves or creates a unique collection for a user.
        """
        collection_name = f"user_{user_id.replace('+', '').replace(':', '')[:20]}"
        return self.client.get_or_create_collection(name=collection_name)
    
    def add_interaction(self, user_id: str, message: str, response: str, metadata: Dict = None):
        """
        Stores an interaction in the user's vector space.
        """
        collection = self.get_user_collection(user_id)
        interaction_text = f"User: {message}\nBot: {response}"
        
        collection.add(
            documents=[interaction_text],
            metadatas=[metadata or {"type": "chat_history"}],
            ids=[str(uuid.uuid4())]
        )
        
    def query_history(self, user_id: str, query: str, n_results: int = 3):
        """
        Searches relevant history for a given query.
        """
        collection = self.get_user_collection(user_id)
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results.get("documents", [[]])[0]

# Singleton
_vector_db_manager = None

def get_vector_db_manager():
    global _vector_db_manager
    if _vector_db_manager is None:
        _vector_db_manager = VectorDBManager()
    return _vector_db_manager
