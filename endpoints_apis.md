---

## ✅ 1. `/chat/parent-timmy`

**Description:**
Handles conversation between **Parent** and **Timmy** (like "Do your homework").
No task analysis is done here — just simple chat.

### 🔹 Method: `POST`

### 🔹 Request Body:

```json
{
  "user_id": "parent_123",
  "message": "Timmy, clean your room."
}
```

### 🔹 Response:

```json
{
  "sender": "Timmy",
  "message": "Okay Mom, I'll clean it."
}
```

### 🔹 What it does:

- Stores the chat in `parent-timmy` ChromaDB collection.
- Retrieves past context (relevant messages).
- Injects them into the chat prompt.
- Returns Timmy’s response.

---

## ✅ 2. `/chat/parent-mrfrench`

**Description:**
Handles conversation between **Parent** and **Mr. French**, which can be:

- Normal chat (small talk).
- **Task commands** like “Add a task for Timmy: Do Math Homework.”

Mr. French will:

- Analyze the message.
- Extract task intent.
- Add/update/delete a task in Supabase.
- Store logs in `mrfrench-logs`.

### 🔹 Method: `POST`

### 🔹 Request Body:

```json
{
  "user_id": "parent_123",
  "message": "Mr. French, add a task for Timmy: Clean the kitchen at 6pm today."
}
```

### 🔹 Response:

```json
{
  "sender": "MrFrench",
  "message": "Noted. I've added the task for Timmy: 'Clean the kitchen' at 6:00 PM today.",
  "analysis": {
    "action": "add",
    "task": "Clean the kitchen",
    "due_time": "18:00",
    "assignee": "Timmy"
  }
}
```

### 🔹 What it does:

- Embeds the message and retrieves relevant memory from `parent-mrfrench`.
- Calls LangGraph/Mr. French Agent.
- Analyzes intent (`add`, `update`, `delete`, `complete`, etc.).
- Updates tasks in Supabase.
- Logs the analysis in `mrfrench-logs`.

---

## ✅ 3. `/chat/timmy-mrfrench`

**Description:**
Conversation between **Timmy** and **Mr. French**.

Timmy can:

- Ask questions
- Ask about pending tasks
- Report completed tasks (e.g., "I finished my homework").

Mr. French:

- Will retrieve tasks.
- Mark them complete or respond accordingly.

### 🔹 Method: `POST`

### 🔹 Request Body:

```json
{
  "user_id": "timmy_456",
  "message": "Mr. French, I completed my math homework."
}
```

### 🔹 Response:

```json
{
  "sender": "MrFrench",
  "message": "Great job, Timmy! I've marked your math homework as complete.",
  "analysis": {
    "action": "complete",
    "task": "math homework",
    "status": "Completed"
  }
}
```

### 🔹 What it does:

- Embeds message into `timmy-mrfrench` ChromaDB.
- Analyzes it using Mr. French's LangGraph.
- Marks tasks complete in Supabase if identified.

---

## ✅ 4. `/tasks`

**Description:**
Internal or Admin endpoint (optional) to **fetch, add, delete tasks** manually.

### 🔹 Method: `GET`

Returns all tasks for a given user or assignee.

#### Query Params:

```
/tasks?assignee=Timmy
```

#### Response:

```json
[
  {
    "id": "task123",
    "task": "Do Math Homework",
    "status": "Pending",
    "due_time": "19:00",
    "reward": "Ice cream"
  }
]
```

---

## ✅ 5. `/analyze` (optional endpoint)

**Description:**
Manually send a message to Mr. French's task analyzer (debug/test purposes).

### 🔹 Method: `POST`

```json
{
  "message": "Add a task for Timmy to clean the dishes by 8pm."
}
```

### 🔹 Response:

```json
{
  "identified_actions": [
    {
      "action": "add",
      "task": "clean the dishes",
      "due_time": "20:00",
      "assignee": "Timmy"
    }
  ]
}
```

---

## 🧠 Memory (ChromaDB Collections Used)

| Collection Name   | Description                                        |
| ----------------- | -------------------------------------------------- |
| `parent-timmy`    | Conversation history between Parent and Timmy      |
| `parent-mrfrench` | Parent ↔ Mr. French conversations                  |
| `timmy-mrfrench`  | Timmy ↔ Mr. French conversations                   |
| `mrfrench-logs`   | Mr. French's internal analysis, task understanding |

---

## 🔧 Notes:

- All endpoints use POST with a JSON body, except `/tasks` which supports GET.
- `user_id` helps identify the speaker and store history properly.
- Mr. French uses LangGraph to decide if message has a task or not.
- Supabase is used for task storage and reward handling.

---

## 🧪 Example Curl Usage

```bash
curl -X POST http://localhost:8000/chat/parent-mrfrench \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "parent_123",
    "message": "Add a task for Timmy to wash the car at 5pm"
  }'
```

---

Let me know if you want me to generate an **OpenAPI spec**, **Postman collection**, or **documentation in Markdown/Swagger format**!

