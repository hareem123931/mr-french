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
from supabase_service import get_tasks, add_task, delete_all_tasks, update_task, get_timmy_zone, update_timmy_zone # Ensure update_task is imported
from chroma_service import add_message_to_history, get_chat_history, delete_all_chroma_data, delete_collection

app = FastAPI()

# Pydantic models for request bodies
class ChatInput(BaseModel):
    user_input: str
    user_type: str # 'Parent' or 'Timmy'

class TimmyZoneUpdate(BaseModel):
    zone: str

def is_placeholder_message(content: str) -> bool:
    """
    Check if a message is just a placeholder (e.g., "Parent", "Mr. French", "Timmy")
    """
    if not content or not content.strip():
        return True
    
    placeholder_texts = ["Parent", "Mr. French", "Timmy"]
    return content.strip() in placeholder_texts

def filter_placeholder_messages(history: list) -> list:
    """
    Filter out placeholder messages from chat history
    """
    return [msg for msg in history if not is_placeholder_message(msg.get("content", ""))]

def analyze_timmy_zone(tasks: list) -> str:
    """
    Analyze Timmy's performance and determine appropriate zone based on task count and status
    """
    if not tasks:
        return "Green"  # No tasks, default to Green
    
    pending_tasks = [task for task in tasks if task.get("is_completed") == "Pending"]
    overdue_tasks = [task for task in tasks if task.get("is_completed") == "Pending" and task.get("Due_Date") == "Today"]
    
    # Red Zone: 5 or more pending tasks OR 3 or more overdue tasks
    if len(pending_tasks) >= 5 or len(overdue_tasks) >= 3:
        return "Red"
    
    # Green Zone: Normal performance (few pending tasks)
    return "Green"

# Define an endpoint for the home page or health check
@app.get("/")
async def read_root():
    return {"message": "Mr. French Conversational AI is running!"}

# --- Chat Endpoint (Unified) ---

@app.post("/chat/{chat_type}")
async def chat_endpoint(
    chat_type: str = Path(..., regex="^(parent-timmy|parent-mrfrench|timmy-mrfrench)$"),
    chat_input: ChatInput = ...
):
    """
    Handles conversational input for different chat types based on chat_type path parameter.
    """
    user_input = chat_input.user_input
    user_type = chat_input.user_type.capitalize() # Ensure 'Parent' or 'Timmy'

    if user_type not in ["Parent", "Timmy"]:
        raise HTTPException(status_code=400, detail="Invalid user_type. Must be 'Parent' or 'Timmy'.")

    if chat_type not in ["parent-timmy", "parent-mrfrench", "timmy-mrfrench"]:
        raise HTTPException(status_code=400, detail="Invalid chat_type. Must be 'parent-timmy', 'parent-mrfrench', or 'timmy-mrfrench'.")

    # Skip processing if input is just a placeholder
    if is_placeholder_message(user_input):
        logger.info(f"Skipping placeholder message: '{user_input}'")
        return {
            "user_type": user_type,
            "user_input": user_input,
            "ai_response": "",
            "message": "Placeholder message ignored"
        }

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
            analysis = final_state["mrfrench_analysis"]["mr_french_analysis"]
            if analysis.get("intent") == "UPDATE_TASK":
                original_task_name = analysis.get("original_task_name")
                updates = analysis.get("updates")
                if original_task_name and updates and updates.get("is_completed") == "Completed":
                    try:
                        logger.info(f"Attempting to update task '{original_task_name}' to 'Completed' in Supabase.")
                        updated_task_db_response = update_task(original_task_name, updates)
                        if not updated_task_db_response:
                            logger.error(f"Failed to find or update task '{original_task_name}' in Supabase.")
                    except Exception as db_e:
                        logger.error(f"Error during Supabase task update for '{original_task_name}': {db_e}", exc_info=True)

        # Auto-analyze and update Timmy's zone after each conversation
        try:
            all_tasks = get_tasks()
            suggested_zone = analyze_timmy_zone(all_tasks)
            current_zone = get_timmy_zone()
            
            # Only auto-update to Red or Green zones (never auto-set Blue)
            if suggested_zone != current_zone and suggested_zone in ["Red", "Green"]:
                update_timmy_zone(suggested_zone)
                logger.info(f"Auto-updated Timmy's zone from {current_zone} to {suggested_zone}")
        except Exception as zone_e:
            logger.error(f"Error during zone analysis: {zone_e}")

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
    Retrieves the full message history for a given chat type, filtered to remove placeholder messages.
    """
    if chat_type not in ["parent-timmy", "parent-mrfrench", "timmy-mrfrench"]:
        raise HTTPException(status_code=400, detail="Invalid chat_type. Must be 'parent-timmy', 'parent-mrfrench', or 'timmy-mrfrench'.")
    
    history = get_chat_history(chat_type, n_results=100)
    # Filter out placeholder messages
    filtered_history = filter_placeholder_messages(history)
    return {"chat_type": chat_type, "history": filtered_history}

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

@app.get("/mrfrench-logs/{log_type}")
async def get_mrfrench_logs_endpoint(log_type: str):
    """
    Retrieves MrFrench logs for a specific chat type.
    """
    valid_log_types = ["parent-timmy", "parent-mrfrench", "timmy-mrfrench"]
    if log_type not in valid_log_types:
        raise HTTPException(status_code=400, detail=f"Invalid log_type. Must be one of {valid_log_types}.")
    
    try:
        logs = get_chat_history(f"mrfrench-logs-{log_type}", n_results=200)
        return {"log_type": log_type, "mrfrench_logs": logs}
    except Exception as e:
        logger.error(f"Failed to retrieve Mr. French logs for {log_type}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve Mr. French logs: {e}")

@app.get("/timmy-zone")
async def get_timmy_zone_endpoint():
    """
    Retrieves Timmy's current zone.
    """
    try:
        zone = get_timmy_zone()
        all_tasks = get_tasks()
        suggested_zone = analyze_timmy_zone(all_tasks)
        
        return {
            "zone": zone,
            "suggested_zone": suggested_zone,
            "zone_analysis": {
                "total_tasks": len(all_tasks),
                "pending_tasks": len([t for t in all_tasks if t.get("is_completed") == "Pending"]),
                "completed_tasks": len([t for t in all_tasks if t.get("is_completed") == "Completed"]),
                "overdue_tasks": len([t for t in all_tasks if t.get("is_completed") == "Pending" and t.get("Due_Date") == "Today"])
            }
        }
    except Exception as e:
        logger.error(f"Failed to retrieve Timmy's zone: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve Timmy's zone: {e}")

@app.post("/timmy-zone")
async def update_timmy_zone_endpoint(zone_update: TimmyZoneUpdate):
    """
    Updates Timmy's zone. Blue Zone can only be set by Parent manually.
    """
    zone = zone_update.zone
    if not zone or zone not in ["Red", "Green", "Blue"]:
        raise HTTPException(status_code=400, detail="Invalid zone. Must be 'Red', 'Green', or 'Blue'.")

    try:
        result = update_timmy_zone(zone)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
            
        logger.info(f"Timmy's zone updated to: {zone}")
        return {"message": f"Timmy's zone has been set to {zone}.", "zone": zone}
    except Exception as e:
        logger.error(f"Failed to update Timmy's zone: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update Timmy's zone: {e}")


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