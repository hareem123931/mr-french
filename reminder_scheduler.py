from apscheduler.schedulers.background import BackgroundScheduler
from supabase_service import get_tasks
from chroma_service import add_message_to_history

def send_due_task_reminders():
    tasks = get_tasks(status="Pending")
    for task in tasks:
        # TODO: Improve logic for "due soon"
        if task.get('Due_Date', '').lower() == "today":
            add_message_to_history("timmy-parent", f"Reminder: You have a pending task: {task['task']}", "system", "Mr. French")

scheduler = BackgroundScheduler()
scheduler.add_job(send_due_task_reminders, 'interval', hours=1)
scheduler.start()