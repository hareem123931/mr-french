# Architecture Document

**Version:** MVP

## 1. Overview

This system simulates conversational exchanges between a parent and a child (Timmy) using
LLM-driven personas. A central agent, Mr. French, monitors these interactions to extract and
manage actionable tasks. Mr. French also has the ability to communicate directly with either the
parent or the child when required, for example, to follow up on incomplete tasks or relay status
updates. The system is designed for extensibility into behavior tracking and automated follow-up
logic.

## 2. Objectives

```
● Simulate realistic parent child dialogues using AI.
● Extract tasks from conversations automatically.
● Allow Mr. French to act as a context-aware mediator, observer, and task manager.
● Support direct interactions between Mr. French and each participant.
● Enable scheduled follow-up conversations to track task completion.
```
## 3. System Components

### 3.1 Parent Node

```
● Generates parent messages using OpenAI based on the last state of the conversation.
● Simulates instructional or concerned dialogue directed at the child.
● Can also direct queries to Mr. French (e.g., asking for task updates).
```
### 3.2 Child Node (Timmy)

```
● Generates child responses using OpenAI, based on prior messages.
● Reflects behavior patterns such as resistance, compliance, or emotional feedback.
```

```
Python
```
```
● Can respond to either the parent or to Mr. French directly (e.g., marking a task
complete).
```
### 3.3 Mr. French

```
● Core AI observer and decision-maker.
● Monitors all conversations between parent and child.
● Extracts task-related instructions and stores structured tasks.
● Responds to direct queries from the parent or child (e.g., “Did Timmy do his
homework?”)
● Can initiate messages to either participant (e.g., follow-ups, clarifications, or updates).
● Maintains separate tone strategies depending on recipient (supportive with Timmy,
professional with parent).
```
### 3.4 Tool Node

```
● Handles structured task operations:
○ Create task
○ Update task status
- Delete the task
○ Fetch task summaries
● Interfaces with storage.
```
## 4. Conversation Flow & LangGraph Design

### 4.1 State Shape

#### {

```
"messages": List[Dict[str, str]],
"tasks": List[Dict[str, Any]],
"current_speaker": "parent" | "child" | "mr_french",
"followup_queue": List[Dict[str, Any]]
}
```
### 4.2 Nodes

```
Node Name Description
```

```
parent_node Produces^ the^ next^ parent^ message^ or^ responses^ to^ Mr^
French/Timmy.
```
```
child_node Produces^ Timmy's^ response^ to^ the^ parent^ or^ Mr.^ French^
```
```
mr_french_node Observes^ conversation^ and^ optionally^ responds^ to^ either^ speaker^
```
```
tool_node Processes^ task-related^ operations^
```
```
followup_schedul
er
```
```
Adds scheduled follow-up actions to queue for Mr. French
```
```
response_node Generates^ messages^ from^ Mr.^ French^ when^ initiating^
communication
```
## 5. Communication Patterns

### 5.1 Standard Flow

1. parent_node sends an instruction.
2. child_node responds.
3. mr_french_node reviews both messages.
4. If a task is detected, tool_node is called to store it.
5. The loop continues for the next turn.

### 5.2 Mr. French Communicating With Parent or Child

```
● Mr. French can send an unsolicited message (e.g., a reminder) by:
○ Being routed directly to response_node as current_speaker =
"mr_french"
○ Message then directed to either child_node or parent_node depending on
context
● Triggered by:
○ Missed deadlines
○ Task completion confirmations
○ Direct questions from parent or child
```
### 5.3 Task Status Updates from Timmy

```
● Timmy can declare a task done through normal conversation.
● mr_french_node evaluates the statement and, if appropriate:
```

```
None
```
```
JSON
```
```
○ Calls tool_node to mark the task as complete.
```
Example:

```
Child: "I finished my homework already."
→ Mr. French: detects intent → updates task
```
## 6. Scheduled Follow-Ups

### Architecture Consideration

```
● followup_scheduler node that:
○ Periodically checks tasks for overdue status.
○ Pushes scheduled reminders to a queue.
○ Routes reminders through Mr. French to the child or parent.
```
### Follow-Up Trigger Logic

```
● Time-based (e.g., task pending for 1 day)
● Status-based (e.g., task marked incomplete)
● Parent-requested (e.g., "Remind Timmy tomorrow")
```
### Follow-Up Flow

1. followup_scheduler identifies overdue task
2. Adds a reminder to followup_queue
3. Next time the graph runs, Mr. French sends a message:
    ○ "Hi Timmy, have you finished your homework yet?"

## 7. Data Model

### Task Object

#### {


```
None
```
```
"task": "Do your homework",
"created_by": "parent",
"status": "pending" | "completed",
"completed_by": "child",
"created_at": "timestamp",
"completed_at": "timestamp"
}
```
## 8. Prompt Engineering Strategy

```
Node Prompt Style
```
```
Parent
Node
```
```
Structured, directive, or concerned
```
```
Child Node Emotionally reactive or evasive (can respond to
both)
Mr. French Neutral and analytical when observing, Supportive or
when communicating
```
## Data Backend

The system uses two separate storage mechanisms for handling structured and unstructured
data:

### 1. ChromaDB (Vector Store)

```
● Purpose : Store and retrieve conversational history for contextual chat.
● Usage :
○ Stores embeddings of all parent, child, and Mr. French messages.
○ Enables Mr. French to reference prior interactions for follow-up context or
long-term behavior tracking.
```

```
● Integration :
○ OpenAI or other embedding models used to generate vectors.
○ ChromaDB queried during Mr. French's processing for relevant memory snippets.
```
### 2. Supabase (Relational Database)

```
● Purpose : Store and manage structured task data.
● Usage :
○ Stores all tasks created by Mr. French from parsed conversations.
○ Tracks task status (pending, completed), creator (parent or child),
timestamps, and other metadata.
● Integration :
○ Tool node interacts with Supabase via REST or client SDKs.
○ Task updates and queries are handled through direct API calls or RPC functions
if needed.
```