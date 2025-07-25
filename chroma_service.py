# timmy_ai_backend/chroma_service.py

import chromadb
import os
from dotenv import load_dotenv
from chromadb.utils import embedding_functions # Import the utilities

import uuid # For generating unique IDs for messages
from datetime import datetime, timezone # For timestamp management

# Load environment variables from .env file
load_dotenv()

# Initialize ChromaDB Persistent Client
# This will create/load a database in the './chroma_db' directory
CHROMA_DB_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Initialize ChromaDB's built-in OpenAIEmbeddingFunction
# It directly uses the OPENAI_API_KEY from environment variables.
# Ensure OPENAI_API_KEY is set in your .env file.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
    print("WARNING: OpenAI API Key not set properly. Using default embeddings.")
    MOCK_MODE = True
    openai_ef = embedding_functions.DefaultEmbeddingFunction()
else:
    MOCK_MODE = False
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-ada-002" # Or your preferred OpenAI embedding model
    )

def get_or_create_collection(collection_name: str):
    """
    Gets an existing ChromaDB collection or creates a new one if it doesn't exist.

    Args:
        collection_name (str): The name of the collection.

    Returns:
        chromadb.api.models.Collection.Collection: The ChromaDB collection object.
    """
    try:
        # Pass the built-in openai_ef instance
        collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)
        return collection
    except Exception as e:
        print(f"Error getting/creating ChromaDB collection '{collection_name}': {e}")
        return None

def add_message_to_history(collection_name: str, message_content: str, role: str, sender: str, metadata: dict = None) -> bool:
    """
    Adds a message to a specific ChromaDB collection.

    Args:
        collection_name (str): The name of the collection (e.g., 'timmy-parent').
        message_content (str): The text content of the message.
        role (str): The role of the message (e.g., 'user', 'assistant', 'system').
        sender (str): The actual sender of the message (e.g., 'Parent', 'Timmy', 'Mr. French').
        metadata (dict, optional): Additional metadata to store with the message.

    Returns:
        bool: True if successful, False otherwise.
    """
    collection = get_or_create_collection(collection_name)
    if not collection:
        return False

    try:
        if metadata is None:
            metadata = {}
        # Add required metadata
        metadata.update({"role": role, "sender": sender, "timestamp": datetime.now(timezone.utc).isoformat()})

        # Generate a unique ID for each message
        message_id = str(uuid.uuid4())

        collection.add(
            documents=[message_content],
            metadatas=[metadata],
            ids=[message_id]
        )
        # Only log for important collections
        if "mrfrench-logs" in collection_name:
            print(f"MrFrench log added: {message_content[:100]}...")
        return True
    except Exception as e:
        print(f"Error adding message to ChromaDB collection '{collection_name}': {e}")
        return False

def get_chat_history(collection_name: str, n_results: int = 100) -> list:
    """
    Retrieves the chat history from a specific ChromaDB collection.

    Args:
        collection_name (str): The name of the collection.
        n_results (int): The maximum number of messages to retrieve.

    Returns:
        list: A list of dictionaries, each representing a message, sorted by timestamp.
              Example: [{"role": "user", "content": "Hello!"}, ...]
    """
    collection = get_or_create_collection(collection_name)
    if not collection:
        return []

    try:
        results = collection.get(
            ids=None, # Get all IDs
            # Removed the empty where={} clause
            limit=n_results,
            include=['documents', 'metadatas']
        )
        # Combine documents and metadatas, then sort by timestamp
        history = []
        # Ensure 'ids' field exists and is not empty before iterating
        if results and results['ids']:
            for i in range(len(results['ids'])):
                message = {
                    "role": results['metadatas'][i].get("role", "user"), # Default to user if role is missing
                    "content": results['documents'][i],
                    "sender": results['metadatas'][i].get("sender", "Unknown"),
                    "timestamp": results['metadatas'][i].get("timestamp", datetime.now(timezone.utc).isoformat()) # Fallback
                }
                history.append(message)

        # Sort history by timestamp to ensure chronological order for LLM context
        history.sort(key=lambda x: x.get("timestamp", ""))

        # Format for LLM consumption (role and content only)
        llm_formatted_history = [{"role": msg['role'], "content": msg['content']} for msg in history]
        return llm_formatted_history
    except Exception as e:
        print(f"Error fetching chat history from ChromaDB collection '{collection_name}': {e}")
        return []

def retrieve_context(collection_name: str, query_text: str, n_results: int = 5) -> list:
    """
    Retrieves relevant contextual messages from a ChromaDB collection based on a query.

    Args:
        collection_name (str): The name of the collection.
        query_text (str): The query string to find relevant messages.
        n_results (int): The number of top-k most relevant messages to retrieve.

    Returns:
        list: A list of dictionaries, each representing a relevant message.
    """
    collection = get_or_create_collection(collection_name)
    if not collection:
        return []

    try:
        # Use query with text input directly
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        context = []
        if results and results['ids'][0]:  # Check if there are results
            for i in range(len(results['ids'][0])):
                message = {
                    "role": results['metadatas'][0][i].get("role", "user"),
                    "content": results['documents'][0][i],
                    "sender": results['metadatas'][0][i].get("sender", "Unknown"),
                    "distance": results['distances'][0][i],
                    "timestamp": results['metadatas'][0][i].get("timestamp", datetime.now(timezone.utc).isoformat())
                }
                context.append(message)
        return context
    except Exception as e:
        print(f"Error retrieving context from ChromaDB collection '{collection_name}': {e}")
        return []

def delete_collection(collection_name: str) -> bool:
    """
    Deletes a ChromaDB collection entirely.

    Args:
        collection_name (str): The name of the collection to delete.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
        return True
    except Exception as e:
        print(f"Error deleting ChromaDB collection '{collection_name}': {e}")
        return False

def delete_all_chroma_data() -> bool:
    """
    Deletes all ChromaDB collections and data.
    Use with extreme caution!

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Get list of all collections
        collections = chroma_client.list_collections()
        
        # Delete each collection
        for collection in collections:
            chroma_client.delete_collection(name=collection.name)
        
        print(f"All ChromaDB data deleted. {len(collections)} collections removed.")
        return True
    except Exception as e:
        print(f"Error deleting all ChromaDB data: {e}")
        return False

def list_collections() -> list:
    """
    Lists all available ChromaDB collections.

    Returns:
        list: List of collection names.
    """
    try:
        collections = chroma_client.list_collections()
        collection_names = [collection.name for collection in collections]
        return collection_names
    except Exception as e:
        print(f"Error listing ChromaDB collections: {e}")
        return []

def get_collection_count(collection_name: str) -> int:
    """
    Gets the number of documents in a ChromaDB collection.

    Args:
        collection_name (str): The name of the collection.

    Returns:
        int: The number of documents in the collection, or -1 if error.
    """
    collection = get_or_create_collection(collection_name)
    if not collection:
        return -1

    try:
        return collection.count()
    except Exception as e:
        print(f"Error getting collection count for '{collection_name}': {e}")
        return -1

# --- Test Block (Run this file directly to test ChromaDB connection and functions) ---
if __name__ == "__main__":
    print("--- Testing ChromaDB Service ---")
    
    # Test adding a message
    test_collection = "test-collection"
    test_message = "Hello, this is a test message."
    result = add_message_to_history(test_collection, test_message, "user", "TestUser")
    if result:
        print(f"✓ Message added to '{test_collection}'")
    else:
        print(f"✗ Failed to add message to '{test_collection}'")
    
    # Test retrieving chat history
    history = get_chat_history(test_collection, n_results=10)
    if history:
        print(f"✓ Retrieved {len(history)} messages from '{test_collection}'")
        for msg in history[-3:]:  # Show last 3 messages
            print(f"  - {msg['role']}: {msg['content'][:50]}...")
    else:
        print(f"✗ No history retrieved from '{test_collection}'")
    
    # Test context retrieval
    context = retrieve_context(test_collection, "test", n_results=3)
    if context:
        print(f"✓ Retrieved {len(context)} contextual messages")
    else:
        print("✗ No contextual messages retrieved")
    
    # Test listing collections
    collections = list_collections()
    print(f"✓ Available collections: {collections}")
    
    # Test collection count
    count = get_collection_count(test_collection)
    print(f"✓ Collection '{test_collection}' has {count} documents")
    
    # Clean up test collection
    delete_collection(test_collection)
    print(f"✓ Test collection '{test_collection}' deleted")
    
    print("--- ChromaDB Service Test Complete ---")