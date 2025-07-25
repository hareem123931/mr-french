import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timezone # For timestamp management
import json # For pretty printing test results
import difflib

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client
# Ensure SUPABASE_URL and SUPABASE_KEY are set in your .env file
SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARNING: Supabase URL and Key not set in .env file. Running in mock mode.")
    MOCK_MODE = True
    supabase = None
else:
    MOCK_MODE = False
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def find_similar_task(task_name: str, threshold: float = 0.8) -> dict:
    """
    Returns the most similar existing task if above threshold, else None.
    """
    all_tasks = get_tasks()
    task_names = [t['task'] for t in all_tasks]
    matches = difflib.get_close_matches(task_name.lower(), [n.lower() for n in task_names], n=1, cutoff=threshold)
    if matches:
        for t in all_tasks:
            if t['task'].lower() == matches[0]:
                return t
    return None

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
    if MOCK_MODE:
        # Mock implementation for testing
        task_data['id'] = 1
        if 'updatedAt' not in task_data:
            task_data['updatedAt'] = datetime.now(timezone.utc).isoformat()
        print(f"MOCK: Adding task: {task_data}")
        return task_data
    
    similar = find_similar_task(task_data['task'])
    if similar:
        return {"error": "This task already exists.", "existing_task": similar}

    try:
        # Ensure updatedAt is set to current UTC time if not provided (Supabase default usually handles this)
        if 'updatedAt' not in task_data:
            task_data['updatedAt'] = datetime.now(timezone.utc).isoformat()

        # Ensure the key for rewards is 'Reward' (singular) for the database interaction
        # If 'Rewards' was passed in by mistake from LLM, convert it.
        if 'Rewards' in task_data:
            task_data['Reward'] = task_data.pop('Rewards') # Change key from 'Rewards' to 'Reward'

        if 'recurrence' not in task_data:
            task_data['recurrence'] = None

        response = supabase.table("tasks").insert(task_data).execute()
        # Supabase client returns a response object; the data is in response.data
        if response.data:
            return response.data[0] # Return the first item if multiple are returned
        else:
            # Check for specific error codes like PGRST204 (missing column)
            error_details = response.count # Supabase client sometimes puts errors here or in .data field for errors
            if hasattr(response, 'data') and response.data and isinstance(response.data, dict) and 'code' in response.data and response.data['code'] == 'PGRST204':
                 print(f"Failed to add task: Schema Mismatch (PGRST204). Column might be missing or mistyped in Supabase. Details: {response.data.get('message', 'No message')}")
                 return {"error": "Schema Mismatch: Column missing or mistyped in Supabase table.", "details": response.data}
            else:
                print(f"Failed to add task: {response.status_code} - {error_details}")
                return {"error": "Failed to add task", "details": error_details}
    except Exception as e:
        print(f"Error adding task: {e}")
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
        # Ensure updatedAt is updated to current UTC time
        updates['updatedAt'] = datetime.now(timezone.utc).isoformat()
        
        # Ensure the key for rewards is 'Reward' (singular) if present in updates
        if 'Rewards' in updates:
            updates['Reward'] = updates.pop('Rewards')

        target_task_id = task_id
        if not target_task_id and task_name:
            # First, find the task by name to get its exact ID
            found_tasks = find_task_by_name(task_name)
            if found_tasks:
                if len(found_tasks) > 1:
                    print(f"Warning: Multiple exact tasks found for name '{task_name}'. Updating the first one found.")
                target_task_id = found_tasks[0]['id']
            else:
                return {"error": f"No task found with name '{task_name}' for update."}
        
        if not target_task_id:
            return {"error": "Could not determine a specific task ID for update."}

        response = supabase.table("tasks").update(updates).eq("id", target_task_id).execute()

        if response.data:
            return response.data[0]
        else:
            print(f"Failed to update task: {response.status_code} - {response.count}")
            return {"error": "Failed to update task", "details": response.count}
    except Exception as e:
        print(f"Error updating task: {e}")
        return {"error": str(e)}

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
                print(f"Warning: Multiple tasks matched '{task_name}'. Deleting all matching tasks.")
            elif not response.data:
                return {"error": f"No task found with name '{task_name}' for deletion."}

        if response.data:
            print(f"Task deleted successfully: {response.data[0]}")
            return {"success": True, "deleted_task": response.data[0]}
        else:
            print(f"Failed to delete task: {response.status_code} - {response.count}")
            return {"error": "Failed to delete task", "details": response.count}
    except Exception as e:
        print(f"Error deleting task: {e}")
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
    if MOCK_MODE:
        # Mock implementation for testing
        mock_tasks = [
            {"id": 1, "task": "Mock Task 1", "is_completed": "Pending", "Due_Date": "2024-01-15", "Due_Time": "17:00", "Reward": "30 mins video games"},
            {"id": 2, "task": "Mock Task 2", "is_completed": "Progress", "Due_Date": "2024-01-16", "Due_Time": "20:00", "Reward": "Extra dessert"}
        ]
        if status:
            return [t for t in mock_tasks if t["is_completed"] == status]
        return mock_tasks
    
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
        print(f"Error fetching tasks: {e}")
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
        # Using .ilike for a robust search that's generally case-insensitive.
        response = supabase.table("tasks").select("*").ilike("task", task_name).execute()
        if response.data:
            # Filter for exact match in Python to ensure precision.
            exact_matches = [
                task for task in response.data
                if task['task'].strip().lower() == task_name.strip().lower()
            ]
            return exact_matches
        else:
            return []
    except Exception as e:
        print(f"Error finding task by name: {e}")
        return []

def delete_all_tasks() -> dict:
    """
    Deletes all tasks from the Supabase 'tasks' table.
    Use with caution!

    Returns:
        dict: Result status.
    """
    try:
        response = supabase.table("tasks").delete().neq("id", 0).execute() # Delete where id != 0 (deletes all)
        if response.data is not None: # Success or empty result
            print(f"All tasks deleted successfully. Deleted count: {len(response.data) if response.data else 'Unknown'}")
            return {"success": True, "deleted_count": len(response.data) if response.data else 0}
        else:
            print(f"Failed to delete all tasks: {response}")
            return {"error": "Failed to delete all tasks", "details": response}
    except Exception as e:
        print(f"Error deleting all tasks: {e}")
        return {"error": str(e)}

def get_timmy_zone() -> str:
    """
    Retrieves Timmy's current zone from the Supabase 'timmy' table.
    
    Returns:
        str: The current zone ('Red', 'Green', or 'Blue'). Defaults to 'Green' if not found.
    """
    if MOCK_MODE:
        return "Green"
    try:
        response = supabase.table("timmy").select("zone").eq("id", 1).execute()
        if response.data and len(response.data) > 0:
            return response.data[0].get("zone", "Green")
        else:
            return "Green"
    except Exception as e:
        print(f"Error retrieving Timmy's zone: {e}")
        return "Green"

def update_timmy_zone(zone: str) -> dict:
    """
    Updates Timmy's zone in the Supabase 'timmy' table.
    
    Args:
        zone (str): The new zone ('Red', 'Green', or 'Blue').
        
    Returns:
        dict: The updated record or error information.
    """
    valid_zones = ["Red", "Green", "Blue"]
    if zone not in valid_zones:
        return {"error": f"Invalid zone '{zone}'. Must be one of {valid_zones}."}
    if MOCK_MODE:
        return {"id": 1, "zone": zone}
    try:
        response = supabase.table("timmy").upsert({"id": 1, "zone": zone}).execute()
        return response.data[0] if response.data else {"error": "Update failed"}
    except Exception as e:
        print(f"Error updating Timmy's zone: {e}")
        return {"error": str(e)}

# --- Test Block (Run this file directly to test Supabase connection and functions) ---
if __name__ == "__main__":
    print("--- Testing Supabase Service ---")

    # Ensure you have your .env variables set up before running tests
    # Add a dummy task
    print("\nAdding a test task...")
    new_task_data = {
        "task": "Test Homework",
        "is_completed": "Pending",
        "Due_Date": "Tomorrow",
        "Due_Time": "Evening",
        "Reward": "Ice Cream" # Corrected to singular 'Reward'
    }
    added_task = add_task(new_task_data)
    if not added_task.get("error"):
        test_task_id = added_task.get("id")
        test_task_name = added_task.get("task")
        print(f"Added task with ID: {test_task_id}")

        # Get all tasks
        print("\nFetching all tasks...")
        all_tasks = get_tasks()
        print(f"All tasks: {json.dumps(all_tasks, indent=2)}")

        # Get pending tasks
        print("\nFetching pending tasks...")
        pending_tasks = get_tasks(status="Pending")
        print(f"Pending tasks: {json.dumps(pending_tasks, indent=2)}")

        # Find task by name
        print(f"\nFinding task by name '{test_task_name}'...")
        found_tasks = find_task_by_name(test_task_name)
        print(f"Found tasks: {json.dumps(found_tasks, indent=2)}")
        if found_tasks and found_tasks[0]['id'] == test_task_id:
            print("find_task_by_name works correctly.")
        else:
            print("find_task_by_name failed or returned unexpected result.")

        # Update the task
        print(f"\nUpdating task '{test_task_name}' to 'Completed'...")
        # Now passing task_name to update_task to simulate the flow from conversation_flow.py
        updated_task = update_task(task_name=test_task_name, updates={"is_completed": "Completed"})
        if not updated_task.get("error") and updated_task.get("is_completed") == "Completed":
            print(f"Task updated successfully to: {updated_task['is_completed']}")
        else:
            print(f"Task update failed: {updated_task.get('error', 'Unknown error')}")

        # Verify update
        print("\nFetching tasks again to verify update...")
        tasks_after_update = get_tasks(status="Completed")
        updated_task_in_db = next((t for t in tasks_after_update if t['id'] == test_task_id), None)
        if updated_task_in_db and updated_task_in_db['is_completed'] == 'Completed':
            print("Update verified in DB.")
        else:
            print("Update not verified in DB.")

        # Delete the task
        print(f"\nDeleting task with ID: {test_task_id}...")
        deleted_result = delete_task(task_id=test_task_id)
        if deleted_result.get("success"):
            print("Task deleted successfully.")
        else:
            print(f"Task deletion failed: {deleted_result.get('error', 'Unknown error')}")

        # Verify deletion
        print("\nFetching all tasks after deletion...")
        remaining_tasks = get_tasks()
        if not any(t['id'] == test_task_id for t in remaining_tasks):
            print("Deletion verified.")
        else:
            print("Deletion not verified.")
    else:
        print("Initial task addition failed, skipping further tests.")

    # Test delete_all_tasks (uncomment with caution, this clears your table)
    print("\n--- Deleting all tasks for final cleanup ---")
    delete_all_tasks_result = delete_all_tasks()
    print(f"Delete all result: {delete_all_tasks_result}")
    print(f"Tasks after mass delete: {get_tasks()}")

    print("\n--- Supabase Service Test Complete ---")