# Project Description: Mr. French Conversational AI

## Introduction

This project is a conversational AI system designed to simulate and manage interactions between a Parent and their child, Timmy, with Mr. French as an intelligent agent. Mr. French observes conversations, extracts actionable tasks, manages reminders, and maintains context across all chat flows. The system is built for extensibility, supporting behavior tracking, reward systems, and proactive notifications.

## Architecture

- **FastAPI** serves as the REST API backend, exposing endpoints for chat, history, tasks, logs, and control operations.
- **ChromaDB** is used as a vector store to persist and retrieve chat history, enabling context-aware responses.
- **Supabase** acts as the structured database for tasks, rewards, and Timmy's zone.
- **OpenAI LLM** powers the personas (Parent, Timmy, Mr. French) and the task analysis logic.
- **APScheduler** is used for periodic reminders and proactive triggers.

## Core Features

### 1. Multi-Persona Chat Flows
- **Parent-Timmy:** Direct chat, no task analysis, just conversational exchange.
- **Parent-MrFrench:** Parent can assign, update, or delete tasks for Timmy via Mr. French, who analyzes intent and manages tasks.
- **Timmy-MrFrench:** Timmy can ask questions, report task completion, or interact for advice.

### 2. Task Management
- Tasks are parsed from conversation using LLM analysis.
- Tasks are stored, updated, and deleted in Supabase.
- Duplicate detection and contextual updates are supported.

### 3. Contextual Awareness
- ChromaDB stores all chat history for context retention.
- Mr. French uses previous messages to maintain memory and context.

### 4. Timmy Zone
- Tracks Timmy's behavioral status (Red, Green, Blue).
- Zone can be updated via API, with logic for parent approval.

### 5. Reminders & Proactive Triggers
- Scheduler sends reminders to Timmy for pending tasks.
- Mr. French notifies Parent if Timmy is unresponsive or has too many pending tasks.

### 6. Reward System & Recurring Tasks
- Rewards can be assigned to tasks and mentioned in reminders.
- Recurring tasks are supported, with daily resets and series deletion.

### 7. Logging & Monitoring
- All analyses and actions by Mr. French are logged in dedicated collections.
- Logs are accessible via API for transparency and debugging.

## API Endpoints

- `/chat/{chat_type}`: Main chat endpoint for all flows.
- `/chat/{chat_type}/history`: Fetches chat history.
- `/parent-timmy/message`: Direct message save and analysis.
- `/tasks`: Fetches tasks, filterable by status.
- `/reset-conversation`: Clears all chat and task data.
- `/mrfrench-logs`: Fetches Mr. French's analysis logs.
- `/timmy-zone`: Gets/sets Timmy's zone.
- `/logs/{chat_type}`: Fetches logs for a specific chat type.

## Data Model

- **Tasks Table (Supabase):**
  - `task`: Task description
  - `is_completed`: Status ("Pending", "Completed", "Progress")
  - `Due_Date`, `Due_Time`: Deadline info
  - `Reward`: Optional reward
  - `updatedAt`: Timestamp

- **ChromaDB Collections:**
  - `parent-timmy`, `parent-mrfrench`, `timmy-mrfrench`: Chat histories
  - `mrfrench-logs`: Mr. French's analyses and thoughts

## Summary (Key Points)

- Realistic multi-persona chat simulation (Parent, Timmy, Mr. French)
- Automatic task extraction, management, and reminders
- Context retention via ChromaDB
- Structured task storage and reward system via Supabase
- Timmy Zone management for behavioral tracking
- RESTful API endpoints for all operations
- Automated reminders and proactive notifications
- Comprehensive logging and monitoring
- Extensible for future features (recurring tasks, advanced triggers)

---
This project provides a robust backend for a conversational AI assistant that can be integrated with any frontend for family task management and behavioral