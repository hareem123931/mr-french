# chroma_service.py

import logging
import os
from datetime import datetime
from typing import List, Dict, Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize Chroma client
client = chromadb.Client()

# Embedding function (OpenAI)
embedding_fn = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))

# Collections used in this app
CHROMA_COLLECTION_NAMES = [
    "parent-timmy",
    "parent-mrfrench",
    "timmy-mrfrench",
    "mrfrench-logs"
]

# Ensure all collections exist
collections = {}
def initialize_chroma_collections():
    for name in CHROMA_COLLECTION_NAMES:
        try:
            collections[name] = client.get_or_create_collection(name=name, embedding_function=embedding_fn)
            logger.info(f"ChromaDB collection '{name}' initialized.")
        except Exception as e:
            logger.error(f"Error ensuring ChromaDB collection '{name}': {e}")

initialize_chroma_collections()


def add_message_to_history(chat_type: str, sender: str, receiver: str, content: str, timestamp: datetime):
    if chat_type not in collections:
        raise ValueError(f"Invalid chat_type: {chat_type}")

    metadata = {
        "chat_type": chat_type,
        "timestamp_iso": timestamp.isoformat(),
        "sender": sender,
        "receiver": receiver
    }

    try:
        collections[chat_type].add(
            documents=[content],
            metadatas=[metadata],
            ids=[f"{chat_type}_{timestamp.timestamp()}"],
        )
        logger.info(f"Added message to '{chat_type}' history from '{sender}': '{content[:20]}...'")
    except Exception as e:
        logger.error(f"Error adding message to ChromaDB collection '{chat_type}': {e}")


def get_chat_history(chat_type: str, k: int = 10) -> List[Dict[str, Any]]:
    if chat_type not in collections:
        raise ValueError(f"Invalid chat_type: {chat_type}")

    try:
        results = collections[chat_type].query(
            query_texts=["recent messages"],
            n_results=k,
        )
        history = []
        for doc, metadata in zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]):
            history.append({
                "role": "user" if metadata.get("sender") == metadata.get("receiver") else metadata.get("sender"),
                "content": doc,
                "metadata": metadata
            })
        return history
    except Exception as e:
        logger.error(f"Error retrieving chat history from ChromaDB collection '{chat_type}': {e}")
        return []


def get_mrfrench_logs(k: int = 15) -> List[Dict[str, Any]]:
    collection_name = "mrfrench-logs"
    if collection_name not in collections:
        logger.warning("Mr. French log collection not found.")
        return []

    try:
        results = collections[collection_name].query(
            query_texts=["mr french analysis"],
            n_results=k,
        )
        logs = []
        for doc, metadata in zip(results.get("documents", [[]])[0], results.get("metadatas", [[]])[0]):
            logs.append({
                "content": doc,
                "metadata": metadata
            })
        return logs
    except Exception as e:
        logger.error(f"Error retrieving Mr. French logs: {e}")
        return []


def delete_collection(name: str):
    try:
        client.delete_collection(name)
        logger.info(f"ChromaDB collection '{name}' deleted.")
    except Exception as e:
        logger.error(f"Error deleting ChromaDB collection '{name}': {e}")


def delete_all_chroma_data():
    for collection_name in CHROMA_COLLECTION_NAMES:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted ChromaDB collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete ChromaDB collection '{collection_name}': {str(e)}")