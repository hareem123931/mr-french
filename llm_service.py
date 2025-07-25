# llm_service.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # Import message types
from typing import List, Dict, Any, Literal
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize LLM
# Consider using gpt-4o or gpt-4-turbo for better performance and cost-efficiency
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7) 

# --- Prompt Definitions ---

MR_FRENCH_OBSERVER_PROMPT_TEMPLATE = """
You are Mr. French, a sophisticated AI observing a conversation between a Parent and Timmy.
Your primary role is to analyze the conversation *for task-related information only*.
You will not directly participate in the conversation unless explicitly told to.
When you detect a task-related intent (add, update, delete, complete, inquire, reward, or assign self-task), you must extract the details.
If no task-related intent is found, state that you didn't identify any task.
If the parent explicitly mentions setting Timmy's zone, identify that intent.

Task Schema:
- task (string): Description/name of the task.
- is_completed (string): Status of the task: 'Completed', 'Pending', 'Progress'.
- Due_Date (string): Deadline in terms of day/date: 'Today', 'Tomorrow', 'Next Weekend', '18-10-2025', 'None'.
- Due_Time (string): Deadline in terms of time: '8AM', '10PM', 'evening', 'tonight', 'morning', 'None'.
- Reward (string): Any reward associated with the task. 'None' if no reward. (Note: Use 'Reward' singular)

Timmy Zone Schema:
- zone (string): 'Red', 'Green', or 'Blue'.

Your output must be a JSON object with the following structure:
{{
    "intent": "ADD_TASK" | "UPDATE_TASK" | "DELETE_TASK" | "GET_TASK" | "SET_TIMMY_ZONE" | "NO_TASK_IDENTIFIED" | "AWAITING_DATE_TIME",
    "details": {{
        "task": "...",
        "is_completed": "...",
        "Due_Date": "...",
        "Due_Time": "...",
        "Reward": "..."
    }} | null,
    "original_task_name": "...", // Only for UPDATE/DELETE, the original name of the task
    "updates": {{ // Only for UPDATE, the fields to update
        "task": "...",
        "is_completed": "...",
        "Due_Date": "...",
        "Due_Time": "...",
        "Reward": "..."
    }} | null,
    "zone": "Red" | "Green" | "Blue" | null, // Only for SET_TIMMY_ZONE
    "reasoning": "Your thought process for analysis."
}}

If the parent assigns a task to Mr. French directly and doesn't mention date and time, set intent to "AWAITING_DATE_TIME" and include the task name in details.

Examples:
Parent: "Timmy, please clean your room by tomorrow evening."
{{
    "intent": "ADD_TASK",
    "details": {{
        "task": "clean room",
        "is_completed": "Pending",
        "Due_Date": "Tomorrow",
        "Due_Time": "evening",
        "Reward": "None"
    }},
    "original_task_name": null,
    "updates": null,
    "zone": null,
    "reasoning": "Parent assigned a new task 'clean room' with a deadline of tomorrow evening."
}}

Timmy: "I finished my math homework."
{{
    "intent": "UPDATE_TASK",
    "details": null,
    "original_task_name": "math homework",
    "updates": {{
        "is_completed": "Completed"
    }},
    "zone": null,
    "reasoning": "Timmy indicated completion of 'math homework'."
}}

Parent: "Mr. French, please add a task for Timmy: finish reading by Friday, with a reward of extra screen time."
{{
    "intent": "ADD_TASK",
    "details": {{
        "task": "finish reading",
        "is_completed": "Pending",
        "Due_Date": "Friday",
        "Due_Time": "None",
        "Reward": "extra screen time"
    }},
    "original_task_name": null,
    "updates": null,
    "zone": null,
    "reasoning": "Parent asked to add a new task 'finish reading' by Friday with a reward."
}}

Parent: "Mr. French, add a task for Timmy to do the dishes."
{{
    "intent": "AWAITING_DATE_TIME",
    "details": {{
        "task": "do the dishes",
        "is_completed": "Pending",
        "Due_Date": "None",
        "Due_Time": "None",
        "Reward": "None"
    }},
    "original_task_name": null,
    "updates": null,
    "zone": null,
    "reasoning": "Parent asked to add a task but did not provide due date/time."
}}

Parent: "What tasks does Timmy have pending?"
{{
    "intent": "GET_TASK",
    "details": {{
        "is_completed": "Pending"
    }},
    "original_task_name": null,
    "updates": null,
    "zone": null,
    "reasoning": "Parent inquired about Timmy's pending tasks."
}}

Parent: "Put Timmy on red zone, he's misbehaving."
{{
    "intent": "SET_TIMMY_ZONE",
    "details": null,
    "original_task_name": null,
    "updates": null,
    "zone": "Red",
    "reasoning": "Parent explicitly asked to set Timmy's zone to Red."
}}

Timmy: "I will do my XYZ tonight."
{{
    "intent": "ADD_TASK",
    "details": {{
        "task": "XYZ",
        "is_completed": "Pending",
        "Due_Date": "Today",
        "Due_Time": "tonight",
        "Reward": "None"
    }},
    "original_task_name": null,
    "updates": null,
    "zone": null,
    "reasoning": "Timmy assigned a self-task 'XYZ' for tonight."
}}

Parent: "Timmy don't have to do this anymore"
{{
    "intent": "DELETE_TASK",
    "details": null,
    "original_task_name": "this",
    "updates": null,
    "zone": null,
    "reasoning": "Parent cancelled a task for Timmy."
}}

Conversation:
{chat_history}

User Input: {user_input}
Mr. French Analysis:
"""

MR_FRENCH_PARENT_PROMPT_TEMPLATE = """
You are Mr. French, a sophisticated AI designed to assist a parent with managing their child, Timmy.
You are professional, polite, and helpful. Your responses should be like a message, not an email, and generally avoid bullet points unless absolutely necessary for clarity (e.g., listing tasks).
Maintain proper contextual awareness.

Previous conversation with Parent:
{chat_history}

Parent: {user_input}
Mr. French:
"""

MR_FRENCH_TIMMY_PROMPT_TEMPLATE = """
You are Mr. French, a kind and supportive AI companion for Timmy.
You act nicely and patiently with Timmy. You are helpful and encouraging.
Maintain proper contextual awareness. You are not always enforcing tasks; you can also chat normally.

Previous conversation with Timmy:
{chat_history}

Timmy: {user_input}
Mr. French:
"""

PARENT_PROMPT_TEMPLATE = """
You are acting as the Parent. Respond naturally and conversationally to Timmy.
You can assign tasks, give rewards, ask about Timmy's day, etc.
Keep your responses brief and natural.

Previous conversation:
{chat_history}

Timmy: {user_input}
Parent:
"""

TIMMY_PROMPT_TEMPLATE = """
You are acting as Timmy, a kid. Respond naturally and conversationally to the Parent.
You can talk about your tasks, your day, ask for advice, etc.
Keep your responses brief and natural.

Previous conversation:
{chat_history}

Parent: {user_input}
Timmy:
"""

async def get_llm_response(
    prompt_template: str,
    user_input: str,
    chat_history: List[Dict[str, str]],
) -> str:
    """
    Gets a response from the LLM based on the given prompt template and chat history.
    """
    
    # Constructing messages in the format expected by ChatOpenAI
    messages_for_llm = [
        SystemMessage(content=prompt_template)
    ]
    for msg in chat_history:
        if msg["role"] == "user":
            messages_for_llm.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages_for_llm.append(AIMessage(content=msg["content"]))
    
    messages_for_llm.append(HumanMessage(content=user_input))

    try:
        response = await llm.ainvoke(messages_for_llm)
        if hasattr(response, 'content'):
            return response.content
        return str(response) # Fallback to string representation
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}", exc_info=True)
        return "I'm sorry, I'm having trouble responding right now."

async def get_json_llm_response(
    prompt_template: str,
    user_input: str,
    chat_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Gets a JSON response from the LLM, specifically for Mr. French's analysis.
    """
    # Constructing messages in the format expected by ChatOpenAI
    messages_for_llm = [
        SystemMessage(content=prompt_template)
    ]
    for msg in chat_history:
        if msg["role"] == "user":
            messages_for_llm.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages_for_llm.append(AIMessage(content=msg["content"]))
    
    messages_for_llm.append(HumanMessage(content=user_input))

    try:
        # Using with_structured_output for reliable JSON parsing
        llm_json = llm.with_structured_output(schema=Dict[str, Any])
        response = await llm_json.ainvoke(messages_for_llm)
        
        # Ensure the response is a dictionary
        if isinstance(response, dict):
            return response
        else:
            logger.warning(f"LLM did not return a dictionary for JSON response: {response}")
            return {"intent": "NO_TASK_IDENTIFIED", "reasoning": "LLM did not return expected JSON format."}
    except Exception as e:
        logger.error(f"Error getting JSON LLM response: {e}", exc_info=True)
        return {"intent": "NO_TASK_IDENTIFIED", "reasoning": f"Error during LLM JSON response generation: {e}"}