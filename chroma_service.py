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
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
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
        print(f"Message added to '{collection_name}' by {sender}: '{message_content[:50]}...'")
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
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                message = {
                    "role": results['metadatas'][0][i].get("role", "user"),
                    "content": results['documents'][0][i],
                    "sender": results['metadatas'][0][i].get("sender", "Unknown"),
                    "timestamp": results['metadatas'][0][i].get("timestamp", "")
                }
                context.append(message)
        # Sort by timestamp to maintain order if context is later used for LLM history
        context.sort(key=lambda x: x.get("timestamp", ""))
        return context
    except Exception as e:
        print(f"Error retrieving context from ChromaDB collection '{collection_name}': {e}")
        return []

def delete_collection(collection_name: str) -> bool:
    """
    Deletes a specific ChromaDB collection.
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
    Deletes all relevant ChromaDB collections.
    Used for the /reset-conversation endpoint.
    """
    collection_names = ["timmy-parent", "timmy-mrfrench", "parent-mrfrench", "mrfrench-logs", "test-chat-history", "test-mrfrench-logs"]
    success = True
    for name in collection_names:
        # Check if collection exists before attempting to delete to avoid "does not exist" errors in logs
        try:
            chroma_client.get_collection(name=name)
            if delete_collection(name):
                pass
            else:
                success = False
        except Exception as e:
            # print(f"Collection '{name}' does not exist, skipping deletion.") # This would be too verbose
            pass # Expected error if collection doesn't exist.

    print("All ChromaDB data deletion process complete.")
    return success

# --- Test Block (Run this file directly to test ChromaDB functions) ---
if __name__ == "__main__":
    print("--- Testing ChromaDB Service ---")
    import json

    # Clean up previous test data if any
    print("\nAttempting to delete all test collections first...")
    delete_all_chroma_data()

    test_collection = "test-chat-history"
    log_collection = "test-mrfrench-logs"

    # Add messages to a test chat history collection
    print(f"\nAdding messages to '{test_collection}'...")
    add_message_to_history(test_collection, "Hi Timmy, did you finish your homework?", "user", "Parent")
    add_message_to_history(test_collection, "Not yet, I'm playing games.", "assistant", "Timmy")
    add_message_to_history(test_collection, "Timmy, focus on your homework now.", "user", "Parent")
    add_message_to_history(test_collection, "Okay, I'll start it soon.", "assistant", "Timmy")

    # Get chat history
    print(f"\nFetching chat history from '{test_collection}'...")
    history = get_chat_history(test_collection)
    print(f"History ({len(history)} messages):")
    print(json.dumps(history, indent=2))
    if len(history) == 4:
        print("History retrieval looks good.")
    else:
        print("History retrieval issue.")

    # Add Mr. French log
    print(f"\nAdding a log to '{log_collection}'...")
    add_message_to_history(log_collection, "Identified task 'homework' as pending.", "system", "Mr. French Analyzer", {"chat_type": "timmy-parent"})
    log_history = get_chat_history(log_collection)
    print(f"Mr. French Logs ({len(log_history)} messages):")
    print(json.dumps(log_history, indent=2))
    if len(log_history) == 1:
        print("Log history retrieval looks good.")
    else:
        print("Log history retrieval issue.")


    # Retrieve context
    print(f"\nRetrieving context for 'homework' from '{test_collection}'...")
    context_messages = retrieve_context(test_collection, "What about homework?", n_results=2)
    print(f"Context messages ({len(context_messages)} found):")
    print(json.dumps(context_messages, indent=2))
    if len(context_messages) > 0 and "homework" in context_messages[0]['content'].lower():
        print("Context retrieval looks good.")
    else:
        print("Context retrieval issue.")

    # Delete test collections
    print("\nDeleting test collections...")
    delete_collection(test_collection)
    delete_collection(log_collection)
    print("ChromaDB Service Test Complete.")