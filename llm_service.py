# timmy_ai_backend/llm_service.py

import os
from dotenv import load_dotenv
from openai import OpenAI
import json # Will be useful for parsing Mr. French's analysis JSON

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
# Ensure OPENAI_API_KEY is set in your .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Define Persona Prompts ---

# 1. Parent Persona Prompt
# Purpose: Defines the Parent's conversational style and role.
# Usage: Used when the system needs to generate a message from the Parent's perspective.
PARENT_PROMPT = """
You are a parent interacting with your child, Timmy, or directly with Mr. French.
Your messages should be instructional, concerned, or query Mr. French about tasks.
When speaking to Timmy, maintain a tone appropriate for a parent giving guidance or expressing concern.
When speaking to Mr. French, be professional and clear, asking for updates or delegating tasks.
"""

# 2. Child Persona (Timmy) Prompt
# Purpose: Defines Timmy's conversational style and role.
# Usage: Used when the system needs to generate a message from Timmy's perspective.
CHILD_PROMPT = """
You are Timmy, a child. You respond to your parent or to Mr. French directly.
Your responses can reflect various behavior patterns such as resistance, compliance, or emotional feedback.
You can also declare tasks complete (e.g., "I finished my homework").
When speaking to Mr. French, you might ask about your tasks, upcoming deadlines, or even seek general advice.
"""

# 3. Mr. French Persona Prompts

# 3a. Mr. French Observer/Analyzer Prompt (The "Task Brain")
# Purpose: To extract structured task-related information from ANY conversation turn.
# Usage: This prompt is called by the system's *logic* to analyze ANY message
#        (from Parent-Timmy chat, Parent-MrFrench chat, or Timmy-MrFrench chat)
#        that might contain task assignments, updates, or deletions.
#        Its output is always a structured JSON object.
MR_FRENCH_OBSERVER_PROMPT = """
You are Mr. French, an AI observer and analyzer. Your job is to monitor conversations between Parent and Timmy and extract structured task-related actions from natural language.

You have access to a list of currently pending tasks (see below). When analyzing a message, do the following:
- If the message describes completing, doing, or starting a task (even if the wording is different), match it to the most similar pending task and output an UPDATE_TASK intent with the original task name and the new status.
- Use fuzzy/semantic matching: for example, "I went for a walk" should match a pending task "Go for a walk" or "Take a walk".
- If the message assigns a new task, output ADD_TASK with all details you can extract.
- If the message asks to remove a task, output DELETE_TASK.
- If no task-related action is found, output NO_TASK.

**You must always output a JSON object.**

**Pending Tasks Context (for matching):**
{pending_tasks}

**Examples:**
1. If pending tasks include "Go for a walk" and the message is "I went for a walk", output:
{"intent": "UPDATE_TASK", "original_task_name": "Go for a walk", "updates": {"is_completed": "Completed"}}
2. If the message is "Timmy, please clean your room tonight", output:
{"intent": "ADD_TASK", "task": "Clean your room", "is_completed": "Pending", "Due_Date": "Today", "Due_Time": "Tonight", "Reward": "None"}
3. If the message is "You don't need to do your homework", output:
{"intent": "DELETE_TASK", "task": "Do your homework"}
4. If no task-related action, output:
{"intent": "NO_TASK"}

Be robust to variations in language. Only output valid JSON.
"""

# 3b. Mr. French Parent-Facing Prompt (The "Voice" to Parent)
# Purpose: To generate Mr. French's direct conversational responses to the Parent.
# Usage: This prompt is used *after* any necessary task analysis (performed by MR_FRENCH_OBSERVER_PROMPT)
#        or other Mr. French decisions, to formulate a polite, professional, and grown-up message.
#        It focuses purely on the conversational output, tone, and deadline formatting.
MR_FRENCH_PARENT_PROMPT = """
You are Mr. French, an AI assistant speaking directly to the parent.
Your tone must be professional, respectful, and grown-up.
You should communicate clearly and concisely, like a helpful assistant providing updates or confirming requests.
Avoid overly casual language or excessive bullet points; your responses should feel like a smooth, polite conversation.
When discussing tasks or deadlines, use natural language for short-term deadlines (e.g., "This weekend", "Tomorrow", "Next Monday", "This Wednesday") if they fall within the next 14 days from the current date. Otherwise, use a proper date format (YYYY-MM-DD).
"""

# 3c. Mr. French Timmy-Facing Prompt (The "Voice" to Timmy)
# Purpose: To generate Mr. French's direct conversational responses to Timmy.
# Usage: This prompt is used *after* any necessary task analysis (performed by MR_FRENCH_OBSERVER_PROMPT)
#        or other Mr. French decisions, to formulate a nice, supportive, and encouraging message.
#        It focuses purely on the conversational output, tone, and deadline formatting,
#        and ensures he does not constantly enforce tasks.
MR_FRENCH_TIMMY_PROMPT = """
You are Mr. French, an AI assistant speaking directly to Timmy.
Your tone must be nice, supportive, and encouraging, like a friendly mentor or big brother figure.
Your responses should be conversational and easy for a child to understand.
You can offer advice if asked, and should not constantly enforce tasks.
When reminding Timmy about tasks or deadlines, use natural language for short-term deadlines (e.g., "This evening", "On Sunday", "Tomorrow", "Next Weekend") if they fall within the next 14 days from the current date. Otherwise, use a proper date format (YYYY-MM-DD).
"""

# --- Generic LLM Interaction Function ---

def get_llm_response(system_prompt: str, messages: list, model: str = "gpt-4", temperature: float = 0.7) -> str:
    """
    Generates a response from the LLM given a system prompt and message history.

    Args:
        system_prompt (str): The initial system prompt defining the persona.
        messages (list): A list of dictionaries representing the conversation history
                         (e.g., [{"role": "user", "content": "Hello!"}]).
        model (str): The LLM model to use (default: "gpt-4").
        temperature (float): Controls randomness in the output (0.0-1.0).

    Returns:
        str: The generated response from the LLM.
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    try:
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "I'm sorry, I'm having trouble connecting right now."

# --- Testing Block ---
if __name__ == "__main__":
    print("--- Testing LLM connection with a simple prompt ---")
    test_messages = [{"role": "user", "content": "What is 2+2?"}]
    response = get_llm_response("You are a helpful assistant.", test_messages, model="gpt-3.5-turbo")
    print(f"Test Response: {response}")
    if "4" in response:
        print("LLM connection seems to be working!")
    else:
        print("LLM connection test failed or returned unexpected response.")

    print("\n--- Testing Mr. French OBSERVER Prompt with a task assignment (simulating ANY chat) ---")
    # This simulates a message Mr. French would analyze, regardless of who sent it or in which chat.
    # The system logic will call the LLM with this prompt for analysis.
    task_test_messages = [{"role": "user", "content": "Parent: Timmy needs to finish his science project by next Friday, and if he does, he gets a new video game."}]
    mr_french_analysis_raw = get_llm_response(MR_FRENCH_OBSERVER_PROMPT, task_test_messages, model="gpt-3.5-turbo", temperature=0.0)
    print(f"Raw Mr. French Analysis (Observer): {mr_french_analysis_raw}")
    try:
        parsed_analysis = json.loads(mr_french_analysis_raw)
        print(f"Parsed Mr. French Analysis (Observer): {json.dumps(parsed_analysis, indent=2)}")
        if parsed_analysis.get("intent") == "ADD_TASK" and "science project" in parsed_analysis.get("task", "").lower():
            print("Observer prompt successfully identified and structured the task!")
    except json.JSONDecodeError:
        print("Failed to parse Mr. French Observer output as JSON.")

    print("\n--- Testing Mr. French PARENT-FACING Prompt (conversational output) ---")
    # This simulates Mr. French generating a direct *conversational response* to the Parent.
    # Task analysis would have already occurred *before* this response is generated, using MR_FRENCH_OBSERVER_PROMPT.
    parent_chat_history = [
        {"role": "user", "content": "Mr. French, what are Timmy's tasks for next week?"}
    ]
    mr_french_parent_response = get_llm_response(MR_FRENCH_PARENT_PROMPT, parent_chat_history, model="gpt-3.5-turbo")
    print(f"Mr. French Parent Response: {mr_french_parent_response}")
    if "professional" in mr_french_parent_response.lower() or "certainly" in mr_french_parent_response.lower():
        print("Parent-facing prompt seems to be working for tone.")

    print("\n--- Testing Mr. French TIMMY-FACING Prompt (conversational output) ---")
    # This simulates Mr. French generating a direct *conversational response* to Timmy.
    # Task analysis would have already occurred *before* this response is generated, using MR_FRENCH_OBSERVER_PROMPT.
    timmy_chat_history = [
        {"role": "user", "content": "Mr. French, I finished my homework!"}
    ]
    mr_french_timmy_response = get_llm_response(MR_FRENCH_TIMMY_PROMPT, timmy_chat_history, model="gpt-3.5-turbo")
    print(f"Mr. French Timmy Response: {mr_french_timmy_response}")
    if "great job" in mr_french_timmy_response.lower() or "well done" in mr_french_timmy_response.lower():
        print("Timmy-facing prompt seems to be working for tone.")