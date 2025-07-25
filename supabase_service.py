# supabase_service.py

import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timezone
import json

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client
SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL and Key must be set in the .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def add_task(task_data: dict) -> dict:
    """
    Adds a new task to the Supabase 'tasks' table.
    The 'id' primary key will be automatically populated by Supabase.

    Args:
        task_data (dict): A dictionary containing task details.
                          Expected keys: 'task', 'is_completed', 'Due_Date', 'Due_Time', 'Reward' (optional).
                          'updatedAt' will be automatically set.

    Returns:
        dict: The inserted task data with the Supabase-generated ID, or an error.
    """
    try:
        if 'updatedAt' not in task_data:
            task_data['updatedAt'] = datetime.now(timezone.utc).isoformat()

        if 'Rewards' in task_data:
            task_data['Reward'] = task_data.pop('Rewards')

        response = supabase.table("tasks").insert(task_data).execute()
        if response.data:
            return response.data[0]
        else:
            error_details = response.count
            if hasattr(response, 'data') and response.data and isinstance(response.data, dict) and 'code' in response.data and response.data['code'] == 'PGRST204':
                 return {"error": "Schema Mismatch: Column missing or mistyped in Supabase table.", "details": response.data}
            else:
                return {"error": "Failed to add task", "details": error_details}
    except Exception as e:
        return {"error": str(e)}

def update_task(task_id: str = None, task_name: str = None, updates: dict = {}) -> dict:
    """
    Updates an existing task in the Supabase 'tasks' table by its ID or name.

    Args:
        task_id (str, optional): The ID of the task to update.
        task_name (str, optional): The name of the task to update (case-insensitive, exact match using find_task_by_name).
        updates (dict): A dictionary of fields to update (e.g., {'is_completed': 'Completed'}).

    Returns:
        dict: The updated task data, or an error.
    """
    if not task_id and not task_name:
        return {"error": "Either task_id or task_name must be provided for update."}
    if not updates:
        return {"error": "No updates provided."}

    try:
        updates['updatedAt'] = datetime.now(timezone.utc).isoformat()
        
        if 'Rewards' in updates:
            updates['Reward'] = updates.pop('Rewards')

        target_task_id = task_id
        if not target_task_id and task_name:
            found_tasks = find_task_by_name(task_name)
            if found_tasks:
                if len(found_tasks) > 1:
                    pass
                target_task_id = found_tasks[0]['id']
            else:
                return {"error": f"No task found with name '{task_name}' for update."}
        
        if not target_task_id:
            return {"error": "Could not determine a specific task ID for update."}

        response = supabase.table("tasks").update(updates).eq("id", target_task_id).execute()

        response = supabase.table("tasks").select("*").eq("task", task_name).execute()
        if response.data:
            task_id = response.data[0]['id'] # Assuming 'id' is your primary key
            # Update the task
            updated_data, count = supabase.table("tasks").update(updates).eq("id", task_id).execute()
            if updated_data:
                logging.info(f"Task '{task_name}' updated successfully: {updated_data}")
                return updated_data
            else:
                logging.warning(f"No task updated for '{task_name}' with updates: {updates}")
                return None
        else:
            logging.warning(f"Task '{task_name}' not found for update.")
            return None
    except Exception as e:
        logging.error(f"Error updating task '{task_name}': {e}", exc_info=True)
        raise # Re-raise to be caught by FastAPI's HTTPException

def delete_task(task_id: str = None, task_name: str = None) -> dict:
    """
    Deletes a task from the Supabase 'tasks' table by its ID or name.

    Args:
        task_id (str, optional): The ID of the task to delete.
        task_name (str, optional): The name of the task to delete (case-insensitive, partial match).

    Returns:
        dict: Confirmation of deletion or an error.
    """
    if not task_id and not task_name:
        return {"error": "Either task_id or task_name must be provided for deletion."}

    try:
        if task_id:
            response = supabase.table("tasks").delete().eq("id", task_id).execute()
        elif task_name:
            response = supabase.table("tasks").delete().ilike("task", f"%{task_name}%").execute()
            if len(response.data) > 1:
                pass
            elif not response.data:
                return {"error": f"No task found with name '{task_name}' for deletion."}

        if response.data:
            return {"success": True, "deleted_task": response.data[0]}
        else:
            return {"error": "Failed to delete task", "details": response.count}
    except Exception as e:
        return {"error": str(e)}

def get_tasks(status: str = None) -> list:
    """
    Fetches tasks from the Supabase 'tasks' table, optionally filtered by status.

    Args:
        status (str, optional): Filter by 'is_completed' status ('Pending', 'Progress', 'Completed').
                                If None, all tasks are returned.

    Returns:
        list: A list of task dictionaries.
    """
    try:
        if status:
            response = supabase.table("tasks").select("*").eq("is_completed", status).order("updatedAt", desc=True).execute()
        else:
            response = supabase.table("tasks").select("*").order("updatedAt", desc=True).execute()

        if response.data:
            return response.data
        else:
            return []
    except Exception as e:
        return []

def find_task_by_name(task_name: str) -> list:
    """
    Finds tasks in the Supabase 'tasks' table by name (case-insensitive, exact match).

    Args:
        task_name (str): The name of the task to search for.

    Returns:
        list: A list of task dictionaries that exactly match the name.
    """
    try:
        response = supabase.table("tasks").select("*").ilike("task", task_name).execute()
        if response.data:
            exact_matches = [
                task for task in response.data
                if task['task'].strip().lower() == task_name.strip().lower()
            ]
            return exact_matches
        else:
            return []
    except Exception as e:
        return []

def delete_all_tasks():
    """
    Deletes all tasks from the Supabase 'tasks' table.
    Used for the /reset-conversation endpoint.
    """
    try:
        response = supabase.table("tasks").delete().gt("updatedAt", "1970-01-01T00:00:00+00:00").execute()
        return {"success": True, "count": response.count}
    except Exception as e:
        return {"error": str(e)}

# --- Test Block (Run this file directly to test Supabase connection and functions) ---
if __name__ == "__main__":
    # Add a dummy task
    new_task_data = {
        "task": "Test Homework",
        "is_completed": "Pending",
        "Due_Date": "Tomorrow",
        "Due_Time": "Evening",
        "Reward": "Ice Cream"
    }
    added_task = add_task(new_task_data)
    if not added_task.get("error"):
        test_task_id = added_task.get("id")
        test_task_name = added_task.get("task")

        # Get all tasks
        all_tasks = get_tasks()

        # Get pending tasks
        pending_tasks = get_tasks(status="Pending")

        # Find task by name
        found_tasks = find_task_by_name(test_task_name)

        # Update the task
        updated_task = update_task(task_name=test_task_name, updates={"is_completed": "Completed"})

        # Verify update
        tasks_after_update = get_tasks(status="Completed")
        updated_task_in_db = next((t for t in tasks_after_update if t['id'] == test_task_id), None)

        # Delete the task
        deleted_result = delete_task(task_id=test_task_id)

        # Verify deletion
        remaining_tasks = get_tasks()
    
    # Test delete_all_tasks (uncomment with caution, this clears your table)
    delete_all_all_tasks_result = delete_all_tasks()