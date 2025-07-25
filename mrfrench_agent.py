# timmy_ai_backend/mrfrench_agent.py

import json
from datetime import datetime, timedelta, timezone
import pytz # For timezone handling if needed for specific date parsing (though we're mostly using UTC now)
import time # For time.sleep in testing

from llm_service import get_llm_response, MR_FRENCH_OBSERVER_PROMPT, MR_FRENCH_PARENT_PROMPT, MR_FRENCH_TIMMY_PROMPT
from supabase_service import add_task, update_task, delete_task, get_tasks, find_task_by_name
from chroma_service import add_message_to_history, get_chat_history, retrieve_context

class MrFrenchAgent:
    """
    MrFrenchAgent encapsulates the core intelligence of Mr. French,
    including task analysis, database interaction, Timmy Zone management,
    and conversational response generation.
    """
    _timmy_zone = "Green" # Default Timmy Zone

    def __init__(self):
        print("MrFrenchAgent initialized.")
        # We can load initial context or state here if needed from DB/files

    @classmethod
    def set_timmy_zone(cls, zone: str):
        """Sets the current Timmy Zone. Valid zones: 'Red', 'Green', 'Blue'."""
        valid_zones = ["Red", "Green", "Blue"]
        if zone in valid_zones:
            cls._timmy_zone = zone
            print(f"Timmy Zone set to: {zone}")
        else:
            print(f"Invalid Timmy Zone: {zone}. Must be one of {valid_zones}.")

    @classmethod
    def get_timmy_zone(cls) -> str:
        """Returns the current Timmy Zone."""
        return cls._timmy_zone

    def analyze_message_for_tasks(self, message_content: str, chat_type: str) -> dict:
        """
        Uses Mr. French's Observer/Analyzer prompt to extract task-related actions from a message.
        Logs the analysis to mrfrench-logs.

        Args:
            message_content (str): The content of the message to analyze.
            chat_type (str): The type of chat ('parent-timmy', 'parent-mrfrench', 'timmy-mrfrench').

        Returns:
            dict: Parsed JSON containing intent and task details, or {'intent': 'NO_TASK'} if parsing fails.
        """
        messages = [{"role": "user", "content": message_content}]
        raw_analysis = get_llm_response(MR_FRENCH_OBSERVER_PROMPT, messages, temperature=0.0)

        log_metadata = {
            "chat_type": chat_type,
            "original_message": message_content,
            "analysis_result": raw_analysis # Store raw LLM output
        }
        add_message_to_history("mrfrench-logs", f"Analyzed message: {message_content[:100]}... Result: {raw_analysis[:100]}...",
                               "system", "Mr. French Analyzer", log_metadata)

        try:
            parsed_analysis = json.loads(raw_analysis)
            # Basic validation of intent
            if "intent" not in parsed_analysis or parsed_analysis["intent"] not in ["ADD_TASK", "UPDATE_TASK", "DELETE_TASK", "NO_TASK"]:
                print(f"Warning: LLM returned invalid intent: {parsed_analysis.get('intent', 'N/A')}")
                return {"intent": "NO_TASK"}
            
            # Harmonize 'Rewards' to 'Reward' from LLM output for consistency with DB
            if "Rewards" in parsed_analysis:
                parsed_analysis["Reward"] = parsed_analysis.pop("Rewards")

            if "updates" in parsed_analysis and "Rewards" in parsed_analysis["updates"]:
                parsed_analysis["updates"]["Reward"] = parsed_analysis["updates"].pop("Rewards")

            return parsed_analysis
        except json.JSONDecodeError:
            print(f"Error parsing Mr. French's analysis JSON: {raw_analysis}")
            return {"intent": "NO_TASK"}

    def handle_task_action(self, task_analysis: dict, chat_type: str, message_sender: str) -> str:
        """
        Executes the identified task action (ADD, UPDATE, DELETE) based on analysis.
        Returns a confirmation message for Mr. French to use in his response (if applicable).
        """
        intent = task_analysis.get("intent")
        response_message = ""

        if intent == "ADD_TASK":
            task_name = task_analysis.get("task")
            if not task_name:
                return "I couldn't identify the task you wanted to add."

            # Check for duplicate tasks (case-insensitive, exact name match)
            # LLM might give slightly different casing, so use find_task_by_name
            existing_tasks = find_task_by_name(task_name)
            if existing_tasks:
                print(f"Task '{task_name}' already exists in the database.")
                # If a task exists, and the new request implies adding (not updating), prompt parent to update
                if task_analysis.get("is_completed") in ["Pending", "Progress"]: # Only suggest update if it's not a completion command
                    return f"The task '{task_name}' already exists. Would you like to update it instead?"
                else:
                    pass 

            due_date = task_analysis.get("Due_Date", "Unknown")
            due_time = task_analysis.get("Due_Time", "Unknown")

            # Special handling for Parent-MrFrench chat if date/time is missing
            if chat_type == "parent-mrfrench" and message_sender == "Parent":
                if due_date == "Unknown" or due_time == "Unknown":
                    return f"I can add '{task_name}'. What is the exact due date and time for it?"

            task_data = {
                "task": task_name,
                "is_completed": task_analysis.get("is_completed", "Pending"),
                "Due_Date": due_date,
                "Due_Time": due_time,
                "Reward": task_analysis.get("Reward", "None") # Use 'Reward' key
            }
            result = add_task(task_data)
            if not result.get("error"):
                response_message = f"Okay, I've added the task: '{task_name}' for Timmy. Due: {self._format_deadline(task_data['Due_Date'], task_data['Due_Time'])}."
                if task_data['Reward'] != 'None':
                    response_message += f" Reward: {task_data['Reward']}."
                # If parent assigned task via MrFrench, notify Timmy
                if chat_type == "parent-mrfrench" and message_sender == "Parent":
                    self.notify_timmy_new_task(task_data)
            else:
                response_message = f"I had trouble adding '{task_name}'. Please try again. Error: {result.get('error')}"

        elif intent == "UPDATE_TASK":
            original_task_name = task_analysis.get("original_task_name")
            updates = task_analysis.get("updates", {})
            if not original_task_name or not updates:
                return "I couldn't identify which task to update or what updates to apply."

            # Find the actual task in DB. Use find_task_by_name for more robustness.
            found_tasks = find_task_by_name(original_task_name)
            if not found_tasks:
                return f"I couldn't find a task named '{original_task_name}' to update."
            
            # Assuming we pick the first relevant match for now. More complex logic needed for multiple.
            task_to_update = found_tasks[0]
            
            result = update_task(task_id=task_to_update["id"], updates=updates)
            if not result.get("error"):
                status_update = updates.get("is_completed")
                if status_update:
                    response_message = f"I've updated '{task_to_update['task']}' to '{status_update}'."
                else:
                    response_message = f"I've updated the task '{task_to_update['task']}' as requested."
            else:
                response_message = f"I had trouble updating '{original_task_name}'. Please try again. Error: {result.get('error')}"

        elif intent == "DELETE_TASK":
            task_to_delete = task_analysis.get("task")
            if not task_to_delete:
                return "I couldn't identify the task you wanted to delete."
            
            # Find the actual task in DB.
            found_tasks = find_task_by_name(task_to_delete)
            if not found_tasks:
                return f"I couldn't find a task named '{task_to_delete}' to delete."
            
            # Assuming we pick the first relevant match for now.
            task_to_delete_obj = found_tasks[0]

            result = delete_task(task_id=task_to_delete_obj["id"])
            if result.get("success"):
                response_message = f"Okay, I've removed the task: '{task_to_delete}'."
            else:
                response_message = f"I had trouble deleting '{task_to_delete}'. Please try again. Error: {result.get('error')}"

        elif intent == "NO_TASK":
            pass
        
        return response_message

    def notify_timmy_new_task(self, task_data: dict):
        """
        Sends a notification message to Timmy about a new task assigned by the Parent via Mr. French.
        This would usually be asynchronously handled or triggered in the main chat flow.
        For now, we'll print a simulated message.
        """
        message_for_timmy = f"Hi Timmy! Your parent just assigned you a new task: '{task_data['task']}'. " \
                            f"It's due {self._format_deadline(task_data['Due_Date'], task_data['Due_Time'])}."
        if task_data.get('Reward') and task_data['Reward'] != 'None':
            message_for_timmy += f" You'll get {task_data['Reward']} for completing it!"
        
        # In a real system, this would queue a message for Timmy's chat endpoint.
        print(f"\n--- Mr. French notifying Timmy ---\n{message_for_timmy}\n----------------------------------\n")
        # Store this message in Timmy-MrFrench chat history
        add_message_to_history("timmy-mrfrench", message_for_timmy, "assistant", "Mr. French")

    def generate_mrfrench_response(self, chat_history: list, persona_prompt: str, current_user_message: str, task_action_message: str = "") -> str:
        """
        Generates Mr. French's conversational response based on the persona and recent context.
        Can include a pre-generated task action confirmation message.

        Args:
            chat_history (list): The list of prior messages in the conversation (role, content).
            persona_prompt (str): The specific Mr. French persona prompt (Parent-facing or Timmy-facing).
            current_user_message (str): The user's last message.
            task_action_message (str): An optional message generated by handle_task_action to include.

        Returns:
            str: Mr. French's generated response.
        """
        # Combine task action message with the current user message for LLM context, if applicable
        full_user_input = current_user_message
        if task_action_message:
            full_user_input = f"{task_action_message}\n\n{current_user_message}"

        # Get relevant context from history using ChromaDB
        # This enhances contextual awareness (point 2, 12, 13 in requirements)
        # We query the collection relevant to the current persona.
        # For simplicity, we'll assume chat_history already contains full relevant context.
        # In a more advanced setup, you'd retrieve based on `persona_prompt` determining collection.
        
        # Add the combined input to the messages for the LLM
        messages_for_llm = chat_history + [{"role": "user", "content": full_user_input}]
        
        return get_llm_response(persona_prompt, messages_for_llm)

    def get_formatted_tasks_for_response(self, status: str = None) -> str:
        """
        Fetches tasks and formats them for Mr. French's conversational responses,
        using natural language for short-term deadlines.
        """
        tasks = get_tasks(status=status)
        if not tasks:
            return "There are no tasks to report at the moment."

        formatted_tasks = []
        for task in tasks:
            task_name = task.get('task', 'Unnamed Task')
            due_date = task.get('Due_Date', 'Unknown')
            due_time = task.get('Due_Time', 'Unknown')
            is_completed = task.get('is_completed', 'Pending')
            reward = task.get('Reward', 'None') # Use 'Reward' key

            # Apply natural language for short-term deadlines
            formatted_deadline = self._format_deadline(due_date, due_time)
            
            task_string = f"- '{task_name}' (Status: {is_completed})"
            if due_date != "Unknown" or due_time != "Unknown":
                task_string += f", Due: {formatted_deadline}"
            if reward != 'None':
                task_string += f", Reward: {reward}"
            
            formatted_tasks.append(task_string)
        
        # Present tasks smoothly, avoiding excessive bullet points if possible based on prompt rules
        return "\n".join(formatted_tasks) # LLM will then rephrase this into smooth text

    def _format_deadline(self, due_date: str, due_time: str) -> str:
        """
        Internal helper to format deadlines using natural language for short-term.
        Handles natural language for days of the week and common time phrases.
        """
        current_time_utc = datetime.now(timezone.utc)
        dt_obj = None

        # 1. Parse Date Component
        if due_date.lower() == "today":
            dt_obj = current_time_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        elif due_date.lower() == "tomorrow":
            dt_obj = (current_time_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif due_date.lower() in ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]:
            day_map = {
                "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6
            }
            target_weekday = day_map[due_date.lower()]
            
            # Calculate days until the next occurrence of the target weekday
            days_until = (target_weekday - current_time_utc.weekday() + 7) % 7

            # If days_until is 0, it means today is the target weekday.
            # We need to decide if it's "this" or "next" based on time.
            # For simplicity for now, if it's "today" by weekday, we assume "this"
            # unless a specific time indicates otherwise, handled by time parsing.
            dt_obj = (current_time_utc + timedelta(days=days_until)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            try:
                # Try to parse as a standard date format (e.g., "YYYY-MM-DD")
                dt_obj = datetime.strptime(due_date, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                # If not a recognized date format, return as is.
                return f"{due_date} {due_time if due_time != 'Unknown' else ''}".strip()

        # 2. Parse Time Component and apply to dt_obj
        if dt_obj: # Only proceed if date parsing was successful
            parsed_time_obj = None
            if due_time.lower() == "evening" or due_time.lower() == "tonight":
                parsed_time_obj = dt_obj.replace(hour=21, minute=0, second=0, microsecond=0) # 9 PM
            elif due_time.lower() == "morning":
                parsed_time_obj = dt_obj.replace(hour=9, minute=0, second=0, microsecond=0) # 9 AM
            elif due_time.lower() == "afternoon":
                parsed_time_obj = dt_obj.replace(hour=14, minute=0, second=0, microsecond=0) # 2 PM
            elif due_time.lower() == "midnight":
                # Midnight of the *current* day refers to the start of the next day technically
                # So if date was "Today" and time "Midnight", it means tomorrow 00:00.
                parsed_time_obj = dt_obj.replace(hour=0, minute=0, second=0, microsecond=0)
                if due_date.lower() == "today": # If date was today, midnight means next day
                    parsed_time_obj += timedelta(days=1)
            elif due_time.lower() == "noon":
                parsed_time_obj = dt_obj.replace(hour=12, minute=0, second=0, microsecond=0) # 12 PM
            elif due_time != "Unknown":
                try:
                    # Try to parse as HH:MM
                    time_part = datetime.strptime(due_time, "%H:%M").time()
                    parsed_time_obj = dt_obj.replace(hour=time_part.hour, minute=time_part.minute, second=0, microsecond=0)
                except ValueError:
                    try:
                        # Then try HH AM/PM
                        time_part = datetime.strptime(due_time.upper().replace("AM", " AM").replace("PM", " PM"), "%I %p").time()
                        parsed_time_obj = dt_obj.replace(hour=time_part.hour, minute=time_part.minute, second=0, microsecond=0)
                    except ValueError:
                        # If time parsing fails, use the date's default (start of day or end of day based on date logic)
                        pass
            
            if parsed_time_obj:
                dt_obj = parsed_time_obj # Apply the parsed time to the datetime object
            # else: dt_obj remains with hour=0, minute=0 unless otherwise specified by date logic (e.g. end of day for "today" etc)

        # 3. Format the combined date and time for display
        if not dt_obj: # If date parsing failed, return original strings
            return f"{due_date} {due_time if due_time != 'Unknown' else ''}".strip()

        today_date = current_time_utc.date()
        due_dt_date = dt_obj.date()

        days_diff = (due_dt_date - today_date).days

        display_date_part = ""
        if days_diff == 0:
            display_date_part = "Today"
        elif days_diff == 1:
            display_date_part = "Tomorrow"
        elif 0 < days_diff <= 7: # Within a week, e.g., "this Monday"
            # If the calculated date for "this X" is today, and the time for the task is already past, then it should be "next X"
            if dt_obj.weekday() == current_time_utc.weekday() and dt_obj.time() < current_time_utc.time():
                display_date_part = f"next {dt_obj.strftime('%A')}"
            else:
                display_date_part = f"this {dt_obj.strftime('%A')}"
        elif 7 < days_diff <= 14: # Next week, e.g., "next Monday"
            display_date_part = f"next {dt_obj.strftime('%A')}"
        else: # Far future, use YYYY-MM-DD
            display_date_part = due_dt_date.strftime("%Y-%m-%d")
        
        display_time_part = ""
        if due_time != "Unknown":
            # If a specific time was parsed, format it
            if parsed_time_obj: # Use the dt_obj which has the correct time part now
                 display_time_part = dt_obj.strftime("%I:%M %p").lstrip('0').replace(' 0', ' ') # e.g., "9:00 AM", "5:30 PM"
            else: # Fallback to original string if not specifically parsed (shouldn't happen much now)
                display_time_part = due_time.lower()


        if display_time_part:
            return f"{display_date_part} at {display_time_part}".strip()
        else:
            return display_date_part.strip()


    # --- Timmy Zone Specific Logic (Conceptual - to be integrated with triggers) ---
    def evaluate_timmy_behavior_for_zone_change(self) -> str:
        """
        Evaluates Timmy's behavior and task completion to suggest a zone change.
        This would involve:
        1. Getting pending/completed tasks (get_tasks).
        2. Analyzing Timmy's chat history for sentiment/responsiveness (requires more advanced NLP).
        For now, this is a placeholder.
        """
        pending_tasks = get_tasks(status="Pending")
        # Placeholder logic: If more than 3 pending tasks, suggest Red Zone (requires Parent approval).
        if len(pending_tasks) >= 5: # As per point 6
            if MrFrenchAgent.get_timmy_zone() == "Green":
                print("Timmy has many pending tasks. Consideration for Red Zone (Parent approval needed).")
                return "Many pending tasks"
        # More complex logic for Blue Zone based on exceptional behavior from chat/task completion
        return "No significant change detected"

    # --- Proactive Trigger Mechanism (Conceptual - to be integrated with cron/main loop) ---
    def check_and_trigger_parent_notification(self) -> str:
        """
        Checks conditions to proactively notify the Parent.
        This function would be called by a periodic job (e.g., cron job).
        """
        trigger_reason = self.evaluate_timmy_behavior_for_zone_change() # Reusing this for simplicity
        if trigger_reason != "No significant change detected":
            message_to_parent = f"Dear Parent, I've observed that Timmy is currently facing a challenge. Reason: {trigger_reason}. Would you like to discuss this?"
            print(f"\n--- Mr. French Proactive Notification to Parent ---\n{message_to_parent}\n-----------------------------------\n")
            # In a real system, this queues a message for Parent's chat endpoint
            add_message_to_history("parent-mrfrench", message_to_parent, "assistant", "Mr. French")
            return message_to_parent
        return "" # No notification sent

    # --- Reminder System (Conceptual - to be integrated with cron/main loop) ---
    def get_tasks_for_reminders(self) -> list:
        """
        Fetches pending tasks that are due today/tonight or near their deadline.
        Also tracks if a reminder has already been sent recently for recurring logic.
        """
        pending_tasks = get_tasks(status="Pending")
        tasks_to_remind = []
        now = datetime.now(timezone.utc)

        for task in pending_tasks:
            due_date_str = task.get("Due_Date")
            due_time_str = task.get("Due_Time")
            task_name = task.get("task")
            
            try:
                # Use _format_deadline internally to get a consistent datetime for comparison
                # This is a bit convoluted; a better approach would be for _format_deadline
                # to return the datetime object directly for internal use.
                # For this step, let's re-parse using the logic from _format_deadline
                
                # Start with date part
                due_dt = None
                if due_date_str.lower() == "today":
                    due_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
                elif due_date_str.lower() == "tomorrow":
                    due_dt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                elif due_date_str.lower() in ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]:
                    day_map = {
                        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                        "friday": 4, "saturday": 5, "sunday": 6
                    }
                    target_weekday = day_map[due_date_str.lower()]
                    days_until = (target_weekday - now.weekday() + 7) % 7
                    due_dt = (now + timedelta(days=days_until)).replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    try: # Try YYYY-MM-DD
                        due_dt = datetime.strptime(due_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=0, minute=0, second=0, microsecond=0)
                    except ValueError:
                        continue # Skip if date cannot be parsed

                # Add time part
                if due_dt and due_time_str and due_time_str != "Unknown":
                    parsed_time_obj = None
                    if due_time_str.lower() == "evening" or due_time_str.lower() == "tonight":
                        parsed_time_obj = due_dt.replace(hour=21, minute=0, second=0, microsecond=0)
                    elif due_time_str.lower() == "morning":
                        parsed_time_obj = due_dt.replace(hour=9, minute=0, second=0, microsecond=0)
                    elif due_time_str.lower() == "afternoon":
                        parsed_time_obj = due_dt.replace(hour=14, minute=0, second=0, microsecond=0)
                    elif due_time_str.lower() == "midnight":
                        parsed_time_obj = due_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                        if due_date_str.lower() == "today":
                            parsed_time_obj += timedelta(days=1)
                    elif due_time_str.lower() == "noon":
                        parsed_time_obj = due_dt.replace(hour=12, minute=0, second=0, microsecond=0)
                    else:
                        try: # Try HH:MM
                            time_part = datetime.strptime(due_time_str, "%H:%M").time()
                            parsed_time_obj = due_dt.replace(hour=time_part.hour, minute=time_part.minute, second=0, microsecond=0)
                        except ValueError:
                            try: # Try HH AM/PM
                                time_part = datetime.strptime(due_time_str.upper().replace("AM", " AM").replace("PM", " PM"), "%I %p").time()
                                parsed_time_obj = due_dt.replace(hour=time_part.hour, minute=time_part.minute, second=0, microsecond=0)
                            except ValueError:
                                pass # Keep date's default time if unparseable
                    
                    if parsed_time_obj:
                        due_dt = parsed_time_obj

                # Define 'near deadline' as within 24 hours for simplicity in testing
                # And also ensure the deadline is in the future.
                time_remaining_seconds = (due_dt - now).total_seconds()
                if 0 < time_remaining_seconds < (24 * 3600):
                    # Add logic here to check last_reminder_sent_at to avoid spamming
                    tasks_to_remind.append(task)
            except Exception as e: # Catch any parsing errors from the inner logic
                print(f"Could not parse date/time for task '{task_name}' for reminder check: {e}")
                continue # Skip task if date parsing fails

        return tasks_to_remind

    def process_recurring_tasks(self):
        """
        Checks for and resets recurring tasks daily if their 'Due_Date' implies recurrence.
        This function should be called daily by a cron job.
        """
        all_tasks = get_tasks() # Get all tasks regardless of status
        # print(f"All tasks for recurring check: {json.dumps(all_tasks, indent=2)}") # Debug print
        today_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current_day_of_week = datetime.now(timezone.utc).strftime("%A") # e.g., "Monday"

        for task in all_tasks:
            task_name = task.get("task")
            due_date_str = task.get("Due_Date")
            is_completed = task.get("is_completed")

            # Simple check for recurring phrases in task name or Due_Date/Time
            # This is a heuristic; more robust would be a dedicated 'recurring' field in DB
            is_daily_recurring = "everyday" in task_name.lower() or "daily" in task_name.lower() or due_date_str.lower() == "daily"
            is_weekly_recurring_on_day = current_day_of_week.lower() in due_date_str.lower() and "every" in due_date_str.lower() # e.g., "every Monday"

            if (is_daily_recurring or is_weekly_recurring_on_day) and is_completed == "Completed":
                print(f"Found completed recurring task '{task_name}'. Resetting to Pending for today.")
                update_task(task_id=task["id"], updates={"is_completed": "Pending"})
            # For tasks that are 'daily' but due_date is like 'Mon-Fri', this logic needs refinement
            # The current approach assumes 'everyday' in task name or 'every Monday' for weekly

# --- Test Block (Run this file directly to test MrFrenchAgent functions) ---
if __name__ == "__main__":
    print("--- Testing MrFrenchAgent ---")
    agent = MrFrenchAgent()

    # Clear all existing tasks and ChromaDB collections before starting
    print("\n--- Clearing all existing data for a clean test run ---")
    from supabase_service import delete_all_tasks
    from chroma_service import delete_all_chroma_data
    delete_all_tasks()
    delete_all_chroma_data()
    print("All data cleared.")
    time.sleep(2) # Give DB a moment to clear, increased sleep for reliability

    # Test Timmy Zone
    print(f"\nInitial Timmy Zone: {MrFrenchAgent.get_timmy_zone()}")
    MrFrenchAgent.set_timmy_zone("Blue")
    print(f"New Timmy Zone: {MrFrenchAgent.get_timmy_zone()}")
    MrFrenchAgent.set_timmy_zone("InvalidZone")
    print(f"Timmy Zone after invalid attempt: {MrFrenchAgent.get_timmy_zone()}")

    # Test Task Analysis and Handling (simulate a Parent-Timmy message)
    print("\n--- Simulating Parent-Timmy conversation for task analysis (ADD) ---")
    parent_message_add = "Timmy, you need to clean your room by tonight. If you do, you get an extra hour of screen time!"
    analysis_add = agent.analyze_message_for_tasks(parent_message_add, "parent-timmy")
    print(f"Analysis result (ADD): {analysis_add}")
    if analysis_add["intent"] == "ADD_TASK":
        task_action_response_add = agent.handle_task_action(analysis_add, "parent-timmy", "Parent")
        print(f"Task action response (ADD): {task_action_response_add}")
    else:
        print("Task analysis did not result in ADD_TASK.")
    time.sleep(1) # Small delay for DB consistency

    # Simulate Timmy completing the task
    print("\n--- Simulating Timmy completing the task (UPDATE) ---")
    timmy_completion_message = "Mr. French, I finished cleaning my room!"
    completion_analysis = agent.analyze_message_for_tasks(timmy_completion_message, "timmy-mrfrench")
    print(f"Completion analysis result: {completion_analysis}")
    if completion_analysis["intent"] == "UPDATE_TASK":
        update_response = agent.handle_task_action(completion_analysis, "timmy-mrfrench", "Timmy")
        print(f"Update response: {update_response}")
    else:
        print("Completion analysis did not result in UPDATE_TASK.")
    time.sleep(1) # Small delay for DB consistency

    # Test Parent-MrFrench conversation for adding task (missing deadline)
    print("\n--- Simulating Parent-MrFrench chat: Add task, no deadline ---")
    parent_msg_no_deadline = "Mr. French, please add a task for Timmy: 'read a book'."
    analysis_no_deadline = agent.analyze_message_for_tasks(parent_msg_no_deadline, "parent-mrfrench")
    print(f"Analysis (no deadline): {analysis_no_deadline}")
    if analysis_no_deadline["intent"] == "ADD_TASK":
        response_needed_deadline = agent.handle_task_action(analysis_no_deadline, "parent-mrfrench", "Parent")
        print(f"Mr. French's expected response (needs deadline): {response_needed_deadline}")
        
        # Now simulate parent providing deadline for "read a book"
        print("\n--- Simulating Parent-MrFrench chat: Providing deadline for 'read a book' ---")
        # In a real LangGraph flow, state would preserve the task and combine.
        # Here, we'll simulate the LLM outputting the full task data now.
        parent_msg_with_deadline_analysis = {
            "intent": "ADD_TASK",
            "task": "read a book",
            "is_completed": "Pending",
            "Due_Date": "Sunday", # LLM might infer "Sunday"
            "Due_Time": "evening",
            "Reward": "None" # Consistent with DB
        }
        response_added = agent.handle_task_action(parent_msg_with_deadline_analysis, "parent-mrfrench", "Parent")
        print(f"Mr. French's response (task added with deadline): {response_added}")
    time.sleep(1) # Small delay for DB consistency
    
    # Test adding a task with YYYY-MM-DD date and specific time
    print("\n--- Simulating Parent-MrFrench chat: Add task with specific date/time ---")
    parent_message_specific_date = "Mr. French, please add 'do laundry' due on 2025-08-01 at 3:30 PM. Reward: new video game."
    analysis_specific_date = agent.analyze_message_for_tasks(parent_message_specific_date, "parent-mrfrench")
    print(f"Analysis (specific date): {analysis_specific_date}")
    if analysis_specific_date["intent"] == "ADD_TASK":
        task_action_response_specific = agent.handle_task_action(analysis_specific_date, "parent-mrfrench", "Parent")
        print(f"Task action response (specific date): {task_action_response_specific}")
    else:
        print("Task analysis did not result in ADD_TASK for specific date.")
    time.sleep(1) # Small delay for DB consistency

    # Test getting formatted tasks
    print("\n--- Getting Formatted Tasks (all) ---")
    formatted_all_tasks = agent.get_formatted_tasks_for_response()
    print(f"Formatted tasks: \n{formatted_all_tasks}")

    print("\n--- Getting Formatted Tasks (pending) ---")
    formatted_pending_tasks = agent.get_formatted_tasks_for_response(status="Pending")
    print(f"Formatted pending tasks: \n{formatted_pending_tasks}")

    # Test Proactive Trigger (will only print if conditions met)
    print("\n--- Testing Proactive Trigger ---")
    agent.check_and_trigger_parent_notification()

    # Test Reminders (will only print if conditions met for pending tasks)
    print("\n--- Testing Reminders ---")
    tasks_for_reminders = agent.get_tasks_for_reminders()
    if tasks_for_reminders:
        print(f"Found {len(tasks_for_reminders)} tasks for reminders:")
        for task in tasks_for_reminders:
            agent.send_task_reminder_to_timmy(task)
    else:
        print("No tasks currently meeting reminder criteria.")

    # Test recurring tasks (simulate a daily check)
    print("\n--- Testing Recurring Tasks ---")
    # Add a recurring task and mark it complete
    # Changed test input to align with process_recurring_tasks logic for "Daily" due date
    recurring_task_add_message = "Mr. French, please add 'Brush teeth' due Daily. Reward: healthy smile."
    recurring_analysis = agent.analyze_message_for_tasks(recurring_task_add_message, "parent-mrfrench")
    # Manually adjust analysis for "Daily" Due_Date if LLM doesn't output it
    if recurring_analysis["intent"] == "ADD_TASK":
        recurring_analysis["task"] = "Brush teeth" # Ensure exact name
        recurring_analysis["Due_Date"] = "Daily" # Force to "Daily" for testing recurring logic
        recurring_analysis["Due_Time"] = "Evening" # Add a time for consistency
        add_result_recurring = agent.handle_task_action(recurring_analysis, "parent-mrfrench", "Parent")
        print(f"Added recurring task response: {add_result_recurring}")
        
        # Assert task was added successfully before proceeding
        if "error" in add_result_recurring:
            print(f"Error adding recurring task, skipping update and recurrence test: {add_result_recurring['error']}")
        else:
            time.sleep(2) # Increased sleep for reliability after adding
            found_brush_task = find_task_by_name("Brush teeth")
            print(f"Found brush task after add: {found_brush_task}") # Debug print
            if found_brush_task:
                # Mark as completed
                update_task(task_id=found_brush_task[0]['id'], updates={"is_completed": "Completed"})
                time.sleep(1) # Give DB time for update
                print(f"Manually marked '{found_brush_task[0]['task']}' as Completed for recurrence test.")
                
                # Now process recurring tasks, which should reset it to Pending
                agent.process_recurring_tasks()
                time.sleep(1) # Give DB time for recurrence update

                # Verify it's pending again
                verified_brush_task = find_task_by_name("Brush teeth")
                if verified_brush_task and verified_brush_task[0]['is_completed'] == 'Pending':
                    print("Recurring task 'Brush teeth' successfully reset to Pending.")
                else:
                    print("Failed to reset recurring task 'Brush teeth' to Pending.")
            else:
                print("Could not find 'Brush teeth' after adding it. Recurring task test failed.")
    else:
        print("Could not add recurring task (initial analysis failed).")

    # Clean up test data (important to keep DB clean for next runs)
    print("\n--- Cleaning up all test tasks ---")
    delete_all_tasks()
    delete_all_chroma_data()
    
    print("\n--- MrFrenchAgent Test Complete ---")