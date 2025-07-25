# main.py
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uvicorn
import json
import logging

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

# Define an endpoint for the home page or health check
@app.get("/")
async def read_root():
    return {"message": "Mr. French Conversational AI is running!"}

# --- Chat Endpoint (Unified) ---

@app.post("/chat")
async def chat_endpoint(chat_input: ChatInput):
    """
    Handles conversational input for different chat types based on user_type.
    Determines chat_type (parent-timmy, parent-mrfrench, timmy-mrfrench) based on speaker.
    """
    user_input = chat_input.user_input
    user_type = chat_input.user_type.capitalize() # Ensure 'Parent' or 'Timmy'

    if user_type not in ["Parent", "Timmy"]:
        raise HTTPException(status_code=400, detail="Invalid user_type. Must be 'Parent' or 'Timmy'.")

    # Determine chat_type based on the conversation flow
    # Assuming Mr. French's involvement implies a direct chat with him
    # And if Parent talks and mentions Timmy or vice-versa, it's Parent-Timmy chat.
    # This logic might need refinement based on your LangGraph's routing.
    chat_type: str
    current_speaker: str
    
    # Simple heuristic to determine chat type based on user_input and user_type
    if "mr. french" in user_input.lower():
        if user_type == "Parent":
            chat_type = "parent-mrfrench"
            current_speaker = "Parent"
        else: # user_type == "Timmy"
            chat_type = "timmy-mrfrench"
            current_speaker = "Timmy"
    elif user_type == "Parent":
        # If Parent talks and doesn't explicitly mention Mr. French, assume Parent-Timmy
        chat_type = "parent-timmy"
        current_speaker = "Parent"
    elif user_type == "Timmy":
        # If Timmy talks and doesn't explicitly mention Mr. French, assume Timmy-MrFrench (as per client req 8, Timmy assigns via MrFrench)
        # Or you might want to consider this as Timmy-Parent if that's a flow you need.
        # For now, sticking to Timmy-MrFrench if Mr. French is the default recipient for Timmy.
        chat_type = "timmy-mrfrench"
        current_speaker = "Timmy"
    else:
        raise HTTPException(status_code=400, detail="Could not determine chat type based on user input and type.")


    logger.info(f"Received chat request: user_type='{user_type}', user_input='{user_input}', inferred_chat_type='{chat_type}'")

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
            # This loop streams updates from the graph.
            # The final_state will contain the complete state after execution.
            final_state = s

        if not final_state:
            raise HTTPException(status_code=500, detail="LangGraph did not return a final state.")

        response_message = "No direct AI response generated for this chat type."
        
        # Extract the AI's response based on the final node of the flow
        if "mrfrench_response" in final_state:
            # For Parent-MrFrench and Timmy-MrFrench direct responses
            messages_list = final_state["mrfrench_response"].get("messages", [])
            assistant_messages = [msg["content"] for msg in messages_list if msg.get("role") == "assistant"]
            if assistant_messages:
                response_message = assistant_messages[-1]
                logger.info(f"Mr. French Response: {response_message}")
            else:
                logger.warning("No assistant message found in mrfrench_response node.")
        elif "child_turn" in final_state:
            # For Parent-Timmy flow where Timmy responds
            messages_list = final_state["child_turn"].get("messages", [])
            assistant_messages = [msg["content"] for msg in messages_list if msg.get("role") == "assistant"]
            if assistant_messages:
                response_message = assistant_messages[-1]
                logger.info(f"Timmy Response: {response_message}")
            else:
                logger.warning("No assistant message found in child_turn node.")
        
        # Log Mr. French's analysis if available (for observer mode or direct interaction)
        if final_state.get("mrfrench_analysis", {}).get("mr_french_analysis"):
            logger.info(f"Mr. French Analysis: {final_state['mrfrench_analysis']['mr_french_analysis']}")
            logger.info(f"Mr. French Task Action Response: {final_state['mrfrench_analysis']['mr_french_task_action_response']}")
            
            # --- DEBUGGING TASK COMPLETION ---
            # Explicitly check for UPDATE_TASK intent and call update_task if needed
            analysis = final_state["mrfrench_analysis"]["mr_french_analysis"]
            if analysis.get("intent") == "UPDATE_TASK":
                original_task_name = analysis.get("original_task_name")
                updates = analysis.get("updates")
                if original_task_name and updates and updates.get("is_completed") == "Completed":
                    try:
                        logger.info(f"Attempting to update task '{original_task_name}' to 'Completed' in Supabase.")
                        # Call update_task explicitly here if your LangGraph doesn't handle it
                        # Make sure your update_task function in supabase_service can handle this
                        # If your LangGraph node already calls it, this might be redundant or for debug validation.
                        updated_task_db_response = update_task(original_task_name, updates)
                        logger.info(f"Supabase update_task response: {updated_task_db_response}")
                        if not updated_task_db_response:
                            logger.error(f"Failed to find or update task '{original_task_name}' in Supabase.")
                    except Exception as db_e:
                        logger.error(f"Error during Supabase task update for '{original_task_name}': {db_e}", exc_info=True)


        return {
            "chat_type": chat_type,
            "user_type": user_type,
            "user_input": user_input,
            "ai_response": response_message,
            "final_state_summary": {
                k: v for k, v in final_state.items() if k not in ["messages"] # Avoid returning full message list if it's large
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