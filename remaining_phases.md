## Revised Project Plan

### **Phase 6: API Endpoints (FastAPI) & Core Logic Refinement**

**Objective:** Implement all required REST API endpoints and refine core task management logic for nuanced interactions.

#### 1. **FastAPI Application Setup (`main.py`)**
- Import necessary modules (FastAPI, your LangGraph app, Supabase client, ChromaDB client, etc.)
- Initialize FastAPI app: `app = FastAPI()`

#### 2. **Implement Chat Endpoints (Req 19)**
- For each chat type (`/chat/parent-timmy`, `/chat/parent-mrfrench`, `/chat/timmy-mrfrench`):
  - **POST Endpoint:**
    - Accepts user input (e.g., `{"message": "..."}`)
    - Retrieves relevant conversation context from the corresponding ChromaDB collection (Req 12)
    - Calls the appropriate persona function (Parent, Child, Mr. French via LangGraph)
    - **Parent-Timmy Chat Specifics (Req 0, 20):** Trigger `mr_french_analysis_node` for background observation; store thoughts in `mrfrench-logs`
    - Stores all new messages in relevant ChromaDB collection
    - **Timmy Notification (Req 18):** If Parent assigns a new task via Mr. French, trigger immediate notification message to Timmy
    - Returns the AI's response
  - **GET Endpoint:**
    - Retrieves and returns full message history from the corresponding ChromaDB collection

#### 3. **Implement Control & Monitoring Endpoints**
- **DELETE `/reset-conversation` (Req 19):**
  - Deletes all data from ChromaDB collections
  - Truncates/deletes rows from the `tasks` table in Supabase
- **GET `/mrfrench-logs` (Req 20):**
  - Accepts optional `chat_type` query param
  - Returns logs from `mrfrench-logs` collection
- **GET `/timmy-zone` (Req 24):**
  - Calls `get_timmy_zone()` and returns current zone
- **POST `/timmy-zone` (Req 24):**
  - Accepts `{"zone": "Red"}` or `{"zone": "Blue"}`
  - For Red Zone, requires Parent permission via conversation
  - Updates Timmy's zone in DB and confirms
- **GET `/tasks` (Req 25):**
  - Accepts `status` query param
  - Calls `get_tasks(status=status)`
  - Returns task list for sidebar

#### 4. **Refine Core Task Management Logic (Req 0, 3, 9, 10, 11, 13, 14, 22)**
- **Task Parsing & Action:** Improve `analyze_message_for_tasks` for detecting ADD/UPDATE/DELETE intents
- **Handling Existing Tasks (Req 10, 11):**
  - Check Supabase for duplicates
  - Respond if task exists; distinguish similar tasks
- **Handling Missing Details (Req 14):**
  - Prompt Parent for missing date/time
  - Update pending task once details are provided
- **Conversational Task Deletion (Req 22):**
  - Detect intent, delete task from Supabase, confirm conversationally
- **Contextual Updates (Req 13):**
  - Update task to 'Completed' if Timmy confirms completion in chat

---

### **Phase 7: Scheduling, Reminders & Proactive Triggers**

**Objective:** Implement automated reminders for Timmy and proactive notifications to the Parent based on predefined triggers.

#### 1. **Scheduler Integration (`scheduler.py` or `main.py`)**
- Use `python-crontab`, `APScheduler`, or simple async loop for testing
- Periodically call `check_and_send_reminders_and_triggers()`

#### 2. **`check_and_send_reminders_and_triggers()` Function (Req 5, 6)**
- **A. Task Reminders for Timmy (Req 5):**
  - Fetch "Pending" tasks
  - Determine if task is due soon
  - Use `last_reminder_at` column to avoid repeat messages
  - Trigger reminder via `mrfrench_response_node`
  - Mention rewards in reminder if applicable (Req 7)
- **B. Proactive Triggers to Parent (Req 6):**
  - **Unresponsive Timmy:** No response post reminder
  - **5+ Pending Tasks:** Send alert
  - **Red Zone Behavior:** Send status update
  - Trigger message to Parent and update `last_parent_notification_at`

---

### **Phase 8: Reward System & Recurring Tasks Implementation**

**Objective:** Fully integrate the reward system and enable recurring task functionality.

#### 1. **Reward System Enhancement (Req 7)**
- Parse reward data from Parent input
- Store in `Rewards` column
- Mention reward when Timmy completes task

#### 2. **Recurring Task Implementation (Req 21)**
- Update schema: `is_recurring`, `recurring_interval`
- Detect phrases like "every day" to parse recurrence
- Set recurrence properties on task creation
- Daily reset logic:
  - Identify completed recurring tasks
  - Reset `is_completed` to `Pending`
- On deletion, remove entire recurring series (Req 22)

---

### **Phase 9: Comprehensive Refinement and Testing**

**Objective:** Conduct thorough testing across all features, refine conversational quality, and ensure system robustness.

#### 1. **Thorough Testing**
- **Automated Tests:** Unit and integration tests
- **Manual Testing:**
  - API endpoints: `POST`, `GET`, `DELETE` (Req 19, 20, 24, 25)
  - Conversational flows: All chat types (Req 0, 1)
  - Task management:
    - Duplicate tasks (Req 10, 11)
    - Missing details prompt (Req 14)
    - Conversational updates & deletions (Req 13, 22)
  - Recurring task behavior (Req 21)
  - Timmy Zone transitions and endpoints (Req 4, 24)
  - Proactive triggers (Req 6)
  - Reminder scheduling and reward mentions (Req 5, 7)
  - Context retention (Req 2, 12)
  - Logs validation (`mrfrench-logs`, Req 20)
  - Deadline rendering (Req 15)
  - Persona consistency (Req 16, 17, 23)

#### 2. **Error Handling**
- Add `try-except` blocks around API, DB, and LLM interactions
- Provide user-friendly error messages

#### 3. **Application Logging**
- Use Python's `logging` module for all operations
- Include API requests, DB activity, LLM calls, errors

