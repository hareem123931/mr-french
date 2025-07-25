# main.py
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uvicorn
import json
import logging
from fastapi import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Assuming these are in your project structure
from conversation_flow import app as langgraph_app # Your compiled LangGraph app
from supabase_service import get_tasks, add_task, delete_all_tasks, update_task # Ensure update_task is imported
from chroma_service import add_message_to_history, get_chat_history, delete_all_chroma_data, delete_collection

app = FastAPI()

# Pydantic models for request bodies
class ChatInput(BaseModel):
    user_input: str
    user_type: str # 'Parent' or 'Timmy'

class TimmyZoneUpdate(BaseModel):
    zone: str

def determine_roles(chat_type: str, user_type: str) -> tuple:
    """
    Returns a tuple (user_role, ai_role) based on chat_type and user_type
    """
    if chat_type == "timmy-mrfrench":
        return "Timmy", "Mr. French"
    elif chat_type == "parent-mrfrench":
        return "Parent", "Mr. French"
    elif chat_type == "parent-timmy":
        return "Parent", "Timmy"
    else:
        return user_type, "Mr. French"

# Define an endpoint for the home page or health check
@app.get("/")
async def read_root():
    return {"message": "Mr. French Conversational AI is running!"}

# --- Chat Endpoint (Unified) ---

@app.post("/chat/{chat_type}")
async def chat_endpoint(
    chat_type: str = Path(..., regex="^(parent-timmy|parent-mrfrench|timmy-mrfrench)$"),
    chat_input: Dict[str, str] = ...
):
    """
    Handles conversational input for different chat types based on chat_type path parameter.
    """
    # For parent-timmy, expect {"user_id", "message"}
    if chat_type == "parent-timmy":
        user_id = chat_input.get("user_id")
        message = chat_input.get("message")
        if not user_id or not message:
            raise HTTPException(status_code=400, detail="Missing 'user_id' or 'message' in request body.")
        user_type = "Parent"
        user_input = message
    else:
        # Use legacy ChatInput model for other chat types
        user_input = chat_input.user_input
        user_type = chat_input.user_type.capitalize() # Ensure 'Parent' or 'Timmy'

    if user_type not in ["Parent", "Timmy"]:
        raise HTTPException(status_code=400, detail="Invalid user_type. Must be 'Parent' or 'Timmy'.")

    if chat_type not in ["parent-timmy", "parent-mrfrench", "timmy-mrfrench"]:
        raise HTTPException(status_code=400, detail="Invalid chat_type. Must be 'parent-timmy', 'parent-mrfrench', or 'timmy-mrfrench'.")

    # Set current_speaker based on user_type
    current_speaker = user_type

    logger.info(f"Received chat request: user_type='{user_type}', user_input='{user_input}', chat_type='{chat_type}'")

    initial_state = {
        "chat_type": chat_type,
        "messages": [],
        "user_input": user_input,
        "mr_french_analysis": {},
        "mr_french_task_action_response": "",
        "current_speaker": current_speaker,
        "recipient": "None" # LangGraph will set this
    }

    try:
        final_state = None
        for s in langgraph_app.stream(initial_state, {'recursion_limit': 10}):
            final_state = s

        if not final_state:
            raise HTTPException(status_code=500, detail="LangGraph did not return a final state.")

        response_message = "No direct AI response generated for this chat type."
        
        if chat_type == "parent-timmy":
            # Always expect Timmy's response in child_turn
            if "child_turn" in final_state:
                messages_list = final_state["child_turn"].get("messages", [])
                assistant_messages = [msg["content"] for msg in messages_list if msg.get("role") == "assistant"]
                if assistant_messages:
                    response_message = assistant_messages[-1]
                    logger.info(f"Timmy Response: {response_message}")
                else:
                    logger.warning("No assistant message found in child_turn node.")
            # Save to ChromaDB
            try:
                add_message_to_history(chat_type, user_input, "user", "Parent")
                add_message_to_history(chat_type, response_message, "assistant", "Timmy")
            except Exception as save_e:
                logger.error(f"Failed to save chat history: {save_e}", exc_info=True)
            # Return API contract
            return {"sender": "Timmy", "message": response_message}
        else:
            # Legacy behavior for other chat types
            if "mrfrench_response" in final_state:
                messages_list = final_state["mrfrench_response"].get("messages", [])
                assistant_messages = [msg["content"] for msg in messages_list if msg.get("role") == "assistant"]
                if assistant_messages:
                    response_message = assistant_messages[-1]
                    logger.info(f"Mr. French Response: {response_message}")
                else:
                    logger.warning("No assistant message found in mrfrench_response node.")
            elif "child_turn" in final_state:
                messages_list = final_state["child_turn"].get("messages", [])
                assistant_messages = [msg["content"] for msg in messages_list if msg.get("role") == "assistant"]
                if assistant_messages:
                    response_message = assistant_messages[-1]
                    logger.info(f"Timmy Response: {response_message}")
                else:
                    logger.warning("No assistant message found in child_turn node.")
            if final_state.get("mrfrench_analysis", {}).get("mr_french_analysis"):
                logger.info(f"Mr. French Analysis: {final_state['mrfrench_analysis']['mr_french_analysis']}")
                logger.info(f"Mr. French Task Action Response: {final_state['mrfrench_analysis']['mr_french_task_action_response']}")
                analysis = final_state["mrfrench_analysis"]["mr_french_analysis"]
                if analysis.get("intent") == "UPDATE_TASK":
                    original_task_name = analysis.get("original_task_name")
                    updates = analysis.get("updates")
                    if original_task_name and updates and updates.get("is_completed") == "Completed":
                        try:
                            logger.info(f"Attempting to update task '{original_task_name}' to 'Completed' in Supabase.")
                            updated_task_db_response = update_task(original_task_name, updates)
                            logger.info(f"Supabase update_task response: {updated_task_db_response}")
                            if not updated_task_db_response:
                                logger.error(f"Failed to find or update task '{original_task_name}' in Supabase.")
                        except Exception as db_e:
                            logger.error(f"Error during Supabase task update for '{original_task_name}': {db_e}", exc_info=True)
            try:
                user_role, ai_role = determine_roles(chat_type, user_type)
                add_message_to_history(chat_type, user_input, user_role, user_type)
                add_message_to_history(chat_type, response_message, ai_role, ai_role)
            except Exception as save_e:
                logger.error(f"Failed to save chat history: {save_e}", exc_info=True)
            return {
                "user_type": user_type,
                "user_input": user_input,
                "ai_response": response_message,
                "final_state_summary": {
                    k: v for k, v in final_state.items() if k not in ["messages"]
                }
            }
    except Exception as e:
        logger.error(f"Error during chat processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during chat processing: {e}")

# --- Chat History Endpoint ---
@app.get("/chat/{chat_type}/history")
async def get_chat_history_endpoint(chat_type: str):
    """
    Retrieves the full message history for a given chat type.
    """
    if chat_type not in ["parent-timmy", "parent-mrfrench", "timmy-mrfrench", "mrfrench-logs"]:
        raise HTTPException(status_code=400, detail="Invalid chat_type. Must be 'parent-timmy', 'parent-mrfrench', 'timmy-mrfrench', or 'mrfrench-logs'.")
    history = get_chat_history(chat_type, n_results=100)
    if chat_type == "parent-timmy":
        # Return as list of {"sender", "message"}
        return [{"sender": msg.get("sender", "Unknown"), "message": msg.get("content", "")}
                for msg in get_chat_history(chat_type, n_results=100)]
    return {"chat_type": chat_type, "history": history}

# --- Control & Monitoring Endpoints ---

@app.delete("/reset-conversation")
async def reset_conversation():
    """
    Clears all data from ChromaDB and Supabase tasks table.
    """
    try:
        logger.info("Resetting all conversation data and tasks.")
        delete_all_chroma_data()
        delete_all_tasks()
        logger.info("All conversation data and tasks have been reset successfully.")
        return {"message": "All conversation data and tasks have been reset."}
    except Exception as e:
        logger.error(f"Failed to reset conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset conversation: {e}")

@app.get("/mrfrench-logs")
async def get_mrfrench_logs_endpoint():
    """
    Retrieves all logs from the mrfrench-logs ChromaDB collection.
    """
    try:
        logs = get_chat_history("mrfrench-logs", n_results=200) # Fetch more for logs
        return {"mrfrench_logs": logs}
    except Exception as e:
        logger.error(f"Failed to retrieve Mr. French logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve Mr. French logs: {e}")

@app.get("/timmy-zone")
async def get_timmy_zone_endpoint():
    """
    Retrieves Timmy's current zone.
    (Placeholder - actual zone storage and retrieval logic needs to be implemented)
    """
    logger.info("Request to get Timmy's zone (placeholder).")
    # This is a placeholder. You'll need actual logic to store and retrieve Timmy's zone.
    # Example: return {"zone": get_timmy_zone_from_db()}
    return {"zone": "Green", "message": "Timmy zone retrieval logic needs full implementation."}

@app.post("/timmy-zone")
async def update_timmy_zone_endpoint(zone_update: TimmyZoneUpdate):
    """
    Updates Timmy's zone. Requires Parent's permission for Red Zone.
    (Placeholder - actual zone storage and update logic needs to be implemented)
    """
    zone = zone_update.zone
    if not zone or zone not in ["Red", "Green", "Blue"]:
        raise HTTPException(status_code=400, detail="Invalid zone. Must be 'Red', 'Green', or 'Blue'.")

    logger.info(f"Request to update Timmy's zone to '{zone}' (placeholder).")
    # Placeholder for actual zone update logic and permission handling
    # You will need to integrate this with MrFrenchAgent's permission flow for Red Zone
    # and actual DB storage for Timmy's zone.
    if zone == "Red":
        # Simulate asking for permission, which would happen in a conversational flow
        return {"message": f"Request to set Timmy's zone to {zone} received. Parent permission for Red Zone required in conversational flow (this endpoint is for direct updates by system, not conversational setting)."}
    
    # Simulate direct update for Green/Blue
    return {"message": f"Timmy's zone set to {zone} (placeholder update). Actual DB update logic pending."}


@app.get("/tasks")
async def get_tasks_endpoint(status: Optional[str] = None):
    """
    Retrieves tasks filtered by completion status.
    """
    try:
        logger.info(f"Request to get tasks with status: {status if status else 'All'}")
        tasks = get_tasks(status=status)
        return {"tasks": tasks}
    except Exception as e:
        logger.error(f"Failed to retrieve tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tasks: {e}")

if __name__ == "__main__":
    uvicorn.run(app, port=8000)