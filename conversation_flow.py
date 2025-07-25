# conversation_flow.py

import os
import json
from typing import Dict, List, Any, TypedDict, Union, Literal, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Assuming these modules exist and are correctly implemented ---
from supabase_service import add_task, update_task, get_tasks
from chroma_service import add_message_to_history, get_chat_history

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# --- State Definition ---
class AgentState(TypedDict):
    chat_type: Literal["parent-timmy", "parent-mrfrench", "timmy-mrfrench"]
    messages: List[Dict[str, str]]
    user_input: str
    mr_french_analysis: Dict[str, Any]
    mr_french_task_action_response: str
    current_speaker: Literal["Parent", "Timmy", "Mr. French"]
    recipient: Literal["Parent", "Timmy", "None"]

# --- Helper Functions ---
def _format_messages_for_llm(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    """Converts internal message format to LangChain BaseMessage format."""
    formatted = []
    for msg in messages:
        if msg["role"] == "user":
            formatted.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            formatted.append(SystemMessage(content=msg["content"]))
    return formatted

def _get_full_context_for_llm(chat_type: str, current_messages: List[Dict[str, str]]) -> List[BaseMessage]:
    """Fetches historical context from ChromaDB and combines with current messages."""
    history_docs = get_chat_history(chat_type, n_results=10)
    
    full_context = []
    for doc_dict in history_docs: 
        speaker = doc_dict.get("metadata", {}).get("speaker")
        content = doc_dict.get("document", doc_dict.get("page_content", ""))

        if speaker and content:
            role = "user" if speaker in ["Parent", "Timmy"] else "assistant"
            full_context.append({"role": role, "content": content})
        
    return _format_messages_for_llm(full_context + current_messages)


# --- Nodes ---

def start_node(state: AgentState) -> AgentState:
    new_messages = [{"role": "user", "content": state["user_input"]}]
    state["messages"].extend(new_messages)
    return state

def parent_turn_node(state: AgentState) -> AgentState:
    chat_type = state["chat_type"]
    user_input = state["user_input"]
    speaker = "Parent"

    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()

    # Dynamically set receiver
    if chat_type == "parent-timmy":
        receiver = "Timmy"
    elif chat_type == "parent-mrfrench":
        receiver = "Mr. French"
    else:
        receiver = "Unknown"

    # Save user message
    add_message_to_history(chat_type, sender=speaker, receiver=receiver, content=user_input, timestamp=timestamp)

    # Add to LangGraph state
    state["messages"].append({"role": "user", "content": user_input})
    state["current_speaker"] = speaker
    return state


def mrfrench_analysis_node(state: AgentState) -> AgentState:
    user_input = state["user_input"]
    chat_type = state["chat_type"]
    
    full_context_for_analysis = _get_full_context_for_llm(chat_type, state["messages"])

    all_tasks = get_tasks() 
    pending_tasks = [task for task in all_tasks if task.get("is_completed") == "Pending"]

    tasks_context = ""
    if pending_tasks:
        tasks_context = "\n\nCurrently Pending Tasks (for context, do not respond about these unless user's input refers to them for update/deletion):"
        for task in pending_tasks:
            tasks_context += f"\n- Task: {task.get('task')}, Status: {task.get('is_completed')}, Due: {task.get('Due_Date')} {task.get('Due_Time')}"

    analysis_prompt = SystemMessage(
        content=f"""You are Mr. French, a sophisticated AI assistant.
        Your primary role is to analyze user messages for task-related intents (ADD_TASK, UPDATE_TASK, DELETE_TASK) or general conversation (NO_TASK_IDENTIFIED).
        You also need to identify 'Timmy Zone' related requests (SET_TIMMY_ZONE_RED, SET_TIMMY_ZONE_BLUE).

        **Task Intent Recognition Details:**
        - For ADD_TASK: Extract 'task' (description), 'is_completed' ('Pending', 'Completed', 'Progress'), 'Due_Date' (e.g., 'Today', 'Tomorrow', 'Next weekend', 'YYYY-MM-DD'), 'Due_Time' (e.g., '8 AM', 'evening', 'tonight'), 'Reward'.
        - For UPDATE_TASK: Identify 'original_task_name' (the existing task name to update) and 'updates' dictionary (e.g., {{'is_completed': 'Completed'}}). **Crucially, recognize completion from phrases like "I finished it", "I'm done", "I already watched it", "I did X", etc. and set 'is_completed' to 'Completed'.** Always try to match the user's statement to a pending task from the context, especially if the user mentions completing something that sounds like a task description.
        - For DELETE_TASK: Identify 'task' (name to delete).
        - For SET_TIMMY_ZONE_RED or SET_TIMMY_ZONE_BLUE: Identify the 'zone' ('Red' or 'Blue').

        **Response Format:**
        Respond *only* with a JSON object. If no task/zone intent is identified, use "NO_TASK_IDENTIFIED".

        **Examples:**
        - ADD_TASK: {{"intent": "ADD_TASK", "task": "Clean your room", "is_completed": "Pending", "Due_Date": "Today", "Due_Time": "Evening", "Reward": "None"}}
        - UPDATE_TASK (completion): {{"intent": "UPDATE_TASK", "original_task_name": "Watch F1 movie", "updates": {{"is_completed": "Completed"}}}}
        - UPDATE_TASK (other update): {{"intent": "UPDATE_TASK", "original_task_name": "Do homework", "updates": {{"Due_Date": "Tomorrow"}}}}
        - DELETE_TASK: {{"intent": "DELETE_TASK", "task": "Take out the trash"}}
        - SET_TIMMY_ZONE_RED: {{"intent": "SET_TIMMY_ZONE_RED", "zone": "Red"}}
        - NO_TASK_IDENTIFIED: {{"intent": "NO_TASK_IDENTIFIED"}}

        {tasks_context}
        """
    )
    
    try:
        analysis_response = llm.invoke([analysis_prompt, HumanMessage(content=user_input)], response_format={"type": "json_object"})
        print(f"DEBUG: Mr. French Analysis LLM Raw Response: {analysis_response.content}") # Debug print
        mr_french_analysis = json.loads(analysis_response.content)
        
        state["mr_french_analysis"]["mr_french_analysis"] = mr_french_analysis
        
        task_action_response = ""

        intent = mr_french_analysis.get("intent")
        
        if intent == "ADD_TASK":
            task_data = {
                "task": mr_french_analysis.get("task"),
                "is_completed": mr_french_analysis.get("is_completed", "Pending"),
                "Due_Date": mr_french_analysis.get("Due_Date", "None"),
                "Due_Time": mr_french_analysis.get("Due_Time", "None"),
                "Reward": mr_french_analysis.get("Reward", "None")
            }
            add_task(task_data) 
            task_action_response = f"Okay, I've added the task: '{task_data['task']}' for Timmy. Due: {task_data['Due_Date']} at {task_data['Due_Time']}."

            if state["chat_type"] == "parent-mrfrench":
                timmy_notification_msg = (
                    f"Hi Timmy! Your parent just assigned you a new task: "
                    f"'{task_data['task']}'. It's due {task_data['Due_Date']} at {task_data['Due_Time']}."
                )
                add_message_to_history("timmy-mrfrench", "Mr. French", "Mr. French", timmy_notification_msg)


        elif intent == "UPDATE_TASK":
            original_task_name = mr_french_analysis.get("original_task_name")
            updates = mr_french_analysis.get("updates")
            if original_task_name and updates:
                updated_db_response = update_task(task_name=original_task_name, updates=updates)
                if updated_db_response:
                    task_action_response = f"I've updated '{original_task_name}'. Its status is now: '{updates.get('is_completed', updated_db_response.get('is_completed'))}'."
                else:
                    task_action_response = f"I tried to update '{original_task_name}' but couldn't find it or apply updates."
            else:
                task_action_response = "I couldn't identify which task to update or what updates to apply."

        elif intent == "DELETE_TASK":
            task_name = mr_french_analysis.get("task")
            if task_name:
                # Assuming delete_task exists in supabase_service.py
                # delete_task(task_name) 
                task_action_response = f"I've removed the task: '{task_name}'."
            else:
                task_action_response = "I couldn't identify which task to delete."

        elif intent in ["SET_TIMMY_ZONE_RED", "SET_TIMMY_ZONE_BLUE"]:
            zone = mr_french_analysis.get("zone")
            task_action_response = f"Request to set Timmy's zone to {zone} detected. This will be handled by the dedicated zone endpoint."
        
        else:
            task_action_response = ""

        state["mr_french_task_action_response"] = task_action_response
        state["current_speaker"] = "Mr. French"

    except json.JSONDecodeError as e:
        state["mr_french_analysis"]["mr_french_analysis"] = {"intent": "NO_TASK_IDENTIFIED", "error": "JSON Decode Error"}
        state["mr_french_task_action_response"] = "I'm having a bit of trouble understanding that. Could you rephrase?"
    except Exception as e:
        state["mr_french_analysis"]["mr_french_analysis"] = {"intent": "NO_TASK_IDENTIFIED", "error": str(e)}
        state["mr_french_task_action_response"] = "I encountered an error during analysis."

    return state


def child_turn_node(state: AgentState) -> AgentState:
    chat_type = state["chat_type"]
    current_user_input = state["user_input"]
    mr_french_analysis = state["mr_french_analysis"].get("mr_french_analysis", {})
    
    full_context_for_timmy = _get_full_context_for_llm(chat_type, [{"role": "user", "content": current_user_input}])

    timmy_system_prompt = SystemMessage(
        content="""You are Timmy, a lively and sometimes a bit cheeky child.
        Respond naturally and briefly to your parent.
        Your responses should reflect a child's personality, including occasional resistance to tasks,
        but also willingness to cooperate or express feelings.
        Maintain an age-appropriate vocabulary and tone.
        """
    )

    intent = mr_french_analysis.get("intent")
    
    if intent == "ADD_TASK":
        task_name = mr_french_analysis.get("task", "a new task")
        timmy_response_prompt = f"Your parent just assigned you '{task_name}'. How do you respond? You can be a bit resistant or ask questions."
    elif intent == "UPDATE_TASK" and mr_french_analysis.get("updates", {}).get("is_completed") == "Completed":
        original_task = mr_french_analysis.get("original_task_name", "a task")
        timmy_response_prompt = f"Your parent noticed you completed '{original_task}'. How do you respond to them?"
    elif intent == "DELETE_TASK":
        task_name = mr_french_analysis.get("task", "a task")
        timmy_response_prompt = f"Your parent just said you don't need to do '{task_name}' anymore. How do you respond?"
    elif intent == "NO_TASK_IDENTIFIED":
        timmy_response_prompt = f"Your parent just said '{current_user_input}'. Respond naturally and briefly. Do not mention tasks unless the parent's actual message was about a task."
    else:
        timmy_response_prompt = f"Your parent just said '{current_user_input}'. Respond naturally and briefly. Do not mention tasks unless the parent's actual message was about a task."

    try:
        messages_for_llm = [timmy_system_prompt] + full_context_for_timmy
        messages_for_llm.append(HumanMessage(content=timmy_response_prompt))

        timmy_llm_response = llm.invoke(messages_for_llm)
        print(f"DEBUG: Timmy LLM Raw Response: {timmy_llm_response.content}")
        timmy_content = timmy_llm_response.content

        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()

        add_message_to_history(chat_type, "Timmy", "Parent", timmy_content, timestamp)
        state["messages"].append({"role": "assistant", "content": timmy_content})
        state["current_speaker"] = "Timmy"

    except Exception as e:
        timmy_content = "Uh oh, I'm not sure how to respond right now."
        state["messages"].append({"role": "assistant", "content": timmy_content})
        state["current_speaker"] = "Timmy"

    return state


def mrfrench_response_node(state: AgentState) -> AgentState:
    chat_type = state["chat_type"]
    speaker = "Mr. French"

    if chat_type == "parent-mrfrench":
        receiver = "Parent"
    elif chat_type == "timmy-mrfrench":
        receiver = "Timmy"
    elif chat_type == "parent-timmy":
        receiver = "Parent"  # Mr. French narrating something back to Parent
    else:
        receiver = "Unknown"

    mrfrench_content = state.get("mrfrench_task_action_response", "")
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()

        analysis = state.get("mrfrench_analysis", {})
        user_input = state.get("user_input", "N/A")

        if analysis:
            log_message = json.dumps({
                "chat_type": chat_type,
                "original_input": user_input,
                "analysis": analysis
            }, indent=2)

            add_message_to_history(
                chat_type="mrfrench-logs",
                sender="Mr. French",
                receiver=receiver,
                content=log_message,
                timestamp=timestamp
            )

    except Exception as e:
        logger.warning(f"Failed to write to mrfrench-logs: {e}")

    return state

# --- Graph Definition ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("start_node", start_node)
workflow.add_node("parent_turn", parent_turn_node)
workflow.add_node("mrfrench_analysis", mrfrench_analysis_node)
workflow.add_node("child_turn", child_turn_node) 
workflow.add_node("mrfrench_response", mrfrench_response_node)

# Set entry point
workflow.set_entry_point("start_node")

# Define edges (transitions between nodes)
workflow.add_edge("start_node", "parent_turn") 
workflow.add_edge("parent_turn", "mrfrench_analysis")

workflow.add_conditional_edges(
    "mrfrench_analysis",
    lambda state: state["chat_type"],
    {
        "parent-timmy": END,             
        "parent-mrfrench": "mrfrench_response",
        "timmy-mrfrench": "mrfrench_response"
    },
)

workflow.add_edge("mrfrench_response", END)


# Compile the graph
app = workflow.compile()

# Example of how to run the graph (for standalone testing, though main.py will use it)
if __name__ == "__main__":
    inputs_parent_timmy_1 = {"user_input": "Hello Timmy, how are you?", "chat_type": "parent-timmy", "current_speaker": "Parent", "messages": [], "mr_french_analysis": {}, "mr_french_task_action_response": "", "recipient": "None"}
    for s in app.stream(inputs_parent_timmy_1, {'recursion_limit': 10}):
        pass

    inputs_parent_mrfrench = {"user_input": "Mr. French, please add a task for Timmy: 'read a book for 30 minutes' due tonight.", "chat_type": "parent-mrfrench", "current_speaker": "Parent", "messages": [], "mr_french_analysis": {}, "mr_french_task_action_response": "", "recipient": "None"}
    for s in app.stream(inputs_parent_mrfrench, {'recursion_limit': 10}):
        pass

    add_task({"task": "Watch F1 movie", "is_completed": "Pending", "Due_Date": "Today", "Due_Time": "10 PM", "Reward": "None"})
    inputs_timmy_mrfrench = {"user_input": "Mr. French, I already watched it.", "chat_type": "timmy-mrfrench", "current_speaker": "Timmy", "messages": [], "mr_french_analysis": {}, "mr_french_task_action_response": "", "recipient": "None"}
    for s in app.stream(inputs_timmy_mrfrench, {'recursion_limit': 10}):
        pass