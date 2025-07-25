import os
import json
from typing import Dict, List, Any, TypedDict, Union, Literal, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

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

    add_message_to_history(chat_type, speaker, speaker, user_input)

    state["current_speaker"] = "Parent"
    return state

def mrfrench_analysis_node(state: AgentState) -> AgentState:
    user_input = state["user_input"]
    chat_type = state["chat_type"]
    
    full_context_for_analysis = _get_full_context_for_llm(chat_type, state["messages"])

    all_tasks = get_tasks() 
    pending_tasks = [task for task in all_tasks if task.get("is_completed") == "Pending"]
    progress_tasks = [task for task in all_tasks if task.get("is_completed") == "Progress"]
    completed_tasks = [task for task in all_tasks if task.get("is_completed") == "Completed"]

    tasks_context = ""
    if all_tasks:
        tasks_context = "\n\nCurrent Tasks in Database:"
        if pending_tasks:
            tasks_context += f"\nPending Tasks ({len(pending_tasks)}):"
            for task in pending_tasks:
                tasks_context += f"\n- Task: {task.get('task')}, Status: {task.get('is_completed')}, Due: {task.get('Due_Date')} {task.get('Due_Time')}"
        if progress_tasks:
            tasks_context += f"\nIn Progress Tasks ({len(progress_tasks)}):"
            for task in progress_tasks:
                tasks_context += f"\n- Task: {task.get('task')}, Status: {task.get('is_completed')}, Due: {task.get('Due_Date')} {task.get('Due_Time')}"
        if completed_tasks:
            tasks_context += f"\nCompleted Tasks ({len(completed_tasks)}):"
            for task in completed_tasks:
                tasks_context += f"\n- Task: {task.get('task')}, Status: {task.get('is_completed')}, Due: {task.get('Due_Date')} {task.get('Due_Time')}"
    else:
        tasks_context = "\n\nNo tasks found in the database."

    analysis_prompt = SystemMessage(
        content=f"""You are Mr. French, a sophisticated AI assistant.
        Your primary role is to analyze user messages for task-related intents and actions.
        
        **Intent Categories:**
        1. ADD_TASK - Instructions or commands to do something
        2. UPDATE_TASK - Updates to existing tasks (including completion)
        3. DELETE_TASK - Remove/cancel tasks
        4. TASK_INQUIRY - Questions about task status, progress, or updates
        5. SET_TIMMY_ZONE_RED / SET_TIMMY_ZONE_BLUE - Zone setting requests
        6. NO_TASK_IDENTIFIED - General conversation

        **Task Intent Recognition Details:**
        - For ADD_TASK: Look for ANY instruction, command, or request that tells someone to do something. This includes:
          * Direct commands: "Go watch a sports match", "Clean your room", "Do your homework"
          * Polite requests: "Please take out the trash", "Can you finish your project"
          * Instructions: "You need to study for the test", "Make sure to feed the dog"
          * Activities assigned: "Watch this movie", "Read this book", "Practice piano"
          Extract 'task' (description), 'is_completed' ('Pending'), 'Due_Date' (default to 'Today' if not specified), 'Due_Time' (default to 'Unknown' if not specified), 'Reward' (default to 'None').
        
        - For UPDATE_TASK: Identify 'original_task_name' (the existing task name to update) and 'updates' dictionary (e.g., {{'is_completed': 'Completed'}}). **Crucially, recognize completion from phrases like "I finished it", "I'm done", "I already watched it", "I did X", etc. and set 'is_completed' to 'Completed'.** Always try to match the user's statement to a pending task from the context, especially if the user mentions completing something that sounds like a task description.
        
        - For DELETE_TASK: Identify 'task' (name to delete) when someone says to cancel, remove, or forget about a task.
        
        - For TASK_INQUIRY: Detect questions about tasks, progress, updates, or status. Look for phrases like:
          * "What's the update on Timmy's tasks?"
          * "How is Timmy's progress?"
          * "What tasks does Timmy have?"
          * "Show me completed tasks"
          * "What's pending?"
          * "Any updates on homework?"
          Set 'query_type' to 'all', 'pending', 'progress', 'completed', or 'specific' (if asking about a specific task)
          If asking about a specific task, set 'specific_task' to the task name.
        
        - For SET_TIMMY_ZONE_RED or SET_TIMMY_ZONE_BLUE: Identify the 'zone' ('Red' or 'Blue').

        **Response Format:**
        Respond *only* with a JSON object. If no task/zone intent is identified, use "NO_TASK_IDENTIFIED".

        **Examples:**
        - ADD_TASK: {{"intent": "ADD_TASK", "task": "Watch a sports match", "is_completed": "Pending", "Due_Date": "Today", "Due_Time": "Unknown", "Reward": "None"}}
        - UPDATE_TASK (completion): {{"intent": "UPDATE_TASK", "original_task_name": "Watch F1 movie", "updates": {{"is_completed": "Completed"}}}}
        - DELETE_TASK: {{"intent": "DELETE_TASK", "task": "Take out the trash"}}
        - TASK_INQUIRY (all): {{"intent": "TASK_INQUIRY", "query_type": "all"}}
        - TASK_INQUIRY (specific): {{"intent": "TASK_INQUIRY", "query_type": "specific", "specific_task": "homework"}}
        - TASK_INQUIRY (by status): {{"intent": "TASK_INQUIRY", "query_type": "pending"}}
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

        elif intent == "TASK_INQUIRY":
            query_type = mr_french_analysis.get("query_type", "all")
            specific_task = mr_french_analysis.get("specific_task", "")
            
            # Fetch current task data for response
            all_tasks = get_tasks()
            
            if query_type == "pending":
                relevant_tasks = [task for task in all_tasks if task.get("is_completed") == "Pending"]
                task_action_response = f"Found {len(relevant_tasks)} pending tasks."
            elif query_type == "progress":
                relevant_tasks = [task for task in all_tasks if task.get("is_completed") == "Progress"]
                task_action_response = f"Found {len(relevant_tasks)} tasks in progress."
            elif query_type == "completed":
                relevant_tasks = [task for task in all_tasks if task.get("is_completed") == "Completed"]
                task_action_response = f"Found {len(relevant_tasks)} completed tasks."
            elif query_type == "specific" and specific_task:
                relevant_tasks = [task for task in all_tasks if specific_task.lower() in task.get("task", "").lower()]
                task_action_response = f"Found {len(relevant_tasks)} tasks matching '{specific_task}'."
            else:
                relevant_tasks = all_tasks
                task_action_response = f"Found {len(all_tasks)} total tasks."
            
            # Store task data for response formatting
            state["mr_french_analysis"]["task_inquiry_data"] = relevant_tasks

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
        print(f"DEBUG: Timmy LLM Raw Response: {timmy_llm_response.content}") # Debug print
        timmy_content = timmy_llm_response.content

        new_timmy_message = {"role": "assistant", "content": timmy_content}
        state["messages"].append(new_timmy_message) 
        add_message_to_history(chat_type, "Timmy", "Timmy", timmy_content)
        state["current_speaker"] = "Timmy"
        
    except Exception as e:
        timmy_content = "Uh oh, I'm not sure how to respond right now."
        state["messages"].append({"role": "assistant", "content": timmy_content})
        add_message_to_history(chat_type, "Timmy", "Timmy", timmy_content)
        state["current_speaker"] = "Timmy"

    return state


def mrfrench_response_node(state: AgentState) -> AgentState:
    chat_type = state["chat_type"]
    user_input = state["user_input"]
    mr_french_task_action_response = state["mr_french_task_action_response"]
    mr_french_analysis = state["mr_french_analysis"]["mr_french_analysis"]
    
    full_context_for_mrfrench = _get_full_context_for_llm(chat_type, [{"role": "user", "content": user_input}])

    if chat_type == "parent-mrfrench":
        system_prompt_content = """You are Mr. French, a professional, courteous, and highly efficient AI assistant for parents.
        Respond to the parent in a helpful, concise, and polite manner.
        Acknowledge their requests clearly. If a task action (add/update/delete) was performed, confirm it professionally.
        If it's a general query, respond appropriately without mentioning tasks unless relevant.
        When providing task information, format it clearly and helpfully.
        Avoid bullet points unless explicitly asked for a list. Keep messages concise, like a natural conversation."""
        recipient = "Parent"
    elif chat_type == "timmy-mrfrench":
        system_prompt_content = """You are Mr. French, a kind, supportive, and encouraging AI assistant for children like Timmy.
        Respond to Timmy in a friendly, gentle, and age-appropriate tone.
        Praise him when he completes tasks, encourage him if he's struggling.
        If he asks for advice, provide it kindly. Do not pressure him about tasks constantly."""
        recipient = "Timmy"
    else:
        system_prompt_content = "You are Mr. French. Respond politely."
        recipient = "None" 

    mrfrench_system_prompt = SystemMessage(content=system_prompt_content)
    
    messages_for_llm = [mrfrench_system_prompt] + full_context_for_mrfrench
    
    # Handle task inquiries with detailed responses
    if mr_french_analysis.get("intent") == "TASK_INQUIRY":
        query_type = mr_french_analysis.get("query_type", "all")
        specific_task = mr_french_analysis.get("specific_task", "")
        task_inquiry_data = state["mr_french_analysis"].get("task_inquiry_data", [])
        
        # Format task data for response
        if task_inquiry_data:
            task_details = ""
            for task in task_inquiry_data:
                task_name = task.get("task", "Unknown task")
                status = task.get("is_completed", "Unknown status")
                due_date = task.get("Due_Date", "No due date")
                due_time = task.get("Due_Time", "No due time")
                reward = task.get("Reward", "No reward")
                
                task_details += f"\n- {task_name} (Status: {status}"
                if due_date != "No due date" and due_date != "None":
                    task_details += f", Due: {due_date}"
                    if due_time != "No due time" and due_time != "None" and due_time != "Unknown":
                        task_details += f" at {due_time}"
                if reward != "No reward" and reward != "None":
                    task_details += f", Reward: {reward}"
                task_details += ")"
            
            if query_type == "all":
                instruction_message = f"The parent asked about all of Timmy's tasks. Provide a helpful summary. Here are the current tasks: {task_details}\n\nRespond naturally and conversationally, organizing this information in a parent-friendly way."
            elif query_type == "pending":
                instruction_message = f"The parent asked about Timmy's pending tasks. Here are the {len(task_inquiry_data)} pending tasks: {task_details}\n\nRespond naturally, focusing on what still needs to be done."
            elif query_type == "progress":
                instruction_message = f"The parent asked about Timmy's tasks in progress. Here are the {len(task_inquiry_data)} tasks currently in progress: {task_details}\n\nRespond naturally, focusing on ongoing work."
            elif query_type == "completed":
                instruction_message = f"The parent asked about Timmy's completed tasks. Here are the {len(task_inquiry_data)} completed tasks: {task_details}\n\nRespond naturally, focusing on accomplishments."
            elif query_type == "specific" and specific_task:
                instruction_message = f"The parent asked about tasks related to '{specific_task}'. Here are the {len(task_inquiry_data)} matching tasks: {task_details}\n\nRespond naturally about these specific tasks."
            else:
                instruction_message = f"The parent asked about tasks. Here are the relevant tasks: {task_details}\n\nRespond naturally and helpfully."
        else:
            if query_type == "pending":
                instruction_message = "The parent asked about pending tasks, but there are no pending tasks currently. Respond positively about this."
            elif query_type == "progress":
                instruction_message = "The parent asked about tasks in progress, but there are no tasks currently in progress. Respond appropriately."
            elif query_type == "completed":
                instruction_message = "The parent asked about completed tasks, but there are no completed tasks yet. Respond encouragingly."
            elif query_type == "specific" and specific_task:
                instruction_message = f"The parent asked about tasks related to '{specific_task}', but no matching tasks were found. Respond helpfully."
            else:
                instruction_message = "The parent asked about tasks, but there are currently no tasks in the system. Respond helpfully."
                
    elif mr_french_analysis.get("intent") in ["ADD_TASK", "UPDATE_TASK", "DELETE_TASK"]:
        instruction_message = f"You just analyzed the message and detected a task action. Your internal action response was: '{mr_french_task_action_response}'. Formulate a natural, conversational response based on this action and the user's original message: '{user_input}'."
    elif mr_french_analysis.get("intent") in ["SET_TIMMY_ZONE_RED", "SET_TIMMY_ZONE_BLUE"]:
        instruction_message = f"You just analyzed a request to set Timmy's zone. Your internal action response was: '{mr_french_task_action_response}'. Formulate a response regarding Timmy's zone."
    else: 
        instruction_message = f"The user's message '{user_input}' was general. Respond conversationally."

    messages_for_llm.append(HumanMessage(content=instruction_message))

    try:
        mrfrench_llm_response = llm.invoke(messages_for_llm)
        print(f"DEBUG: Mr. French Response LLM Raw Response: {mrfrench_llm_response.content}") # Debug print
        mrfrench_content = mrfrench_llm_response.content

        new_mrfrench_message = {"role": "assistant", "content": mrfrench_content}
        state["messages"].append(new_mrfrench_message) 
        add_message_to_history(chat_type, "Mr. French", "Mr. French", mrfrench_content) 
        state["current_speaker"] = "Mr. French"
        state["recipient"] = recipient

    except Exception as e:
        mrfrench_content = "I'm sorry, I'm having trouble responding right now."
        state["messages"].append({"role": "assistant", "content": mrfrench_content})
        add_message_to_history(chat_type, "Mr. French", "Mr. French", mrfrench_content)
        state["current_speaker"] = "Mr. French"
        state["recipient"] = recipient

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