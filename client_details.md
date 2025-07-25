0. The Parent can assign tasks to Timmy directly or by conversing with MrFrench. For example, the Parent might say “Add a task for Timmy: XYZ.” MrFrench should recognize when a task is added, updated, or deleted, and act accordingly. He can also engage in casual conversation with the Parent—behaving professionally and courteously—whenever the Parent wishes.

1. We should maintain separate ChromaDB collections for each interaction type:

   * timmy‑parent.
   * timmy‑mrfrench.
   * parent‑mrfrench.
   * mrfrench‑logs (to store MrFrench’s analyses and “thoughts”).

2. All message flows must support proper contextual awareness. For example, if the Parent says “Hi, my name is X” and later asks “What’s my name?”, MrFrench should recall it. Likewise, when Timmy speaks to MrFrench, MrFrench must remember Timmy’s earlier messages.

3. MrFrench’s primary task is to analyze and manage tasks. Whenever the Parent or Timmy chats about something unrelated to tasks, MrFrench should respond normally. But if they refer to tasks, he should leverage his task‑analysis capabilities.

4. **Timmy Zone**, based on Timmy’s behavior and task performance:

   * **Red Zone.** Timmy isn’t completing tasks or is misbehaving. MrFrench can ask the Parent for permission to set Red Zone and then update Timmy’s zone accordingly.
   * **Green Zone.** Timmy is generally doing well; occasional missed tasks are acceptable.
   * **Blue Zone.** Timmy does something exceptionally positive. Only the Parent can instruct MrFrench to set Blue Zone.

5. Implement scheduling and reminders for Timmy’s pending tasks using a cron job. Whenever a task is due today or soon, MrFrench should remind Timmy to start working on it.

6. Develop trigger mechanisms between MrFrench and the Parent. For example, if Timmy isn’t responsive, has five or more pending tasks, or shows poor behavior, MrFrench should notify the Parent with an update on Timmy’s status.

7. Design and integrate a reward system. When the Parent tells Timmy “If you do X by Y, you’ll get Z,” MrFrench should record the reward (optionally) and include it with the task.

8. Timmy can assign tasks to himself via MrFrench (e.g., “I will do XYZ tonight”). MrFrench should add or update tasks based on Timmy’s requests and interact kindly, given Timmy’s age.

9. Supabase schema for the `tasks` table:

   * **task** (text): description or name of the task.
   * **is\_completed** (text): status (“Completed,” “Pending,” “Progress”).
   * **Due\_Date** (text): deadline in words or dates (“Today,” “Tomorrow,” “Next weekend,” “2025‑10‑18,” etc.).
   * **Due\_Time** (text): time (“8 AM,” “10 PM,” “evening,” “tonight,” “morning,” etc.).
   * **updatedAt** (UTC timestamp).
   * **Rewards** (text): any reward assigned by the Parent.

10. In parent‑MrFrench updates/tests: if the Parent assigns a task that already exists, MrFrench should recognize it and say “This task already exists” rather than “Didn’t identify the task.” He must still distinguish between similar but distinct tasks (e.g., “math homework,” “math project,” “math hackathon”).

11. In parent‑Timmy task analyses: apply the same behavior as above—identify existing tasks and report “This task already exists” without adding duplicates, while distinguishing similar tasks correctly.

12. Every chat message should include the full previous conversation for context.

13. Chats must maintain contextual awareness. For example, if MrFrench reminds Timmy “You haven’t completed XYZ” and Timmy replies “I completed it,” MrFrench should mark XYZ as completed in the database.

14. In parent‑MrFrench chat: if the Parent assigns a task without specifying date or time, MrFrench should ask for those details. Once the Parent provides them, MrFrench should add the task with its proper deadline.

15. When the Parent asks about Timmy’s tasks or deadlines, MrFrench should respond conversationally, using relative phrases for near deadlines (“this weekend,” “next Monday,” “Friday”) and exact dates for distant deadlines.

16. Timmy and MrFrench can discuss anything. If Timmy seeks advice, MrFrench should provide it; he shouldn’t pressure Timmy about tasks all night.

17. The Parent and MrFrench can also talk about any topic. But if the Parent inquires about Timmy’s updates, MrFrench should then report on or prompt for new assignments.

18. Parent‑MrFrench messaging: when the Parent assigns a new task through MrFrench, MrFrench should immediately notify Timmy: “You have a new task: XYZ.”

19. Provide REST endpoints for every operation: POST and GET for each chat flow, plus a “reset conversation” endpoint that clears all ChromaDB and Supabase data.

20. Implement “MrFrench logs” for each chat flow. In parent‑Timmy chats, MrFrench acts as an observer—his thought process (task identification, updates, deletions, or “no tasks identified”) must be exposed via a GET endpoint. Apply the same logging in timmy‑MrFrench and parent‑MrFrench flows.

21. Support recurring tasks (e.g., “Do XYZ every day” or “If you do X for Y days, you’ll get Z reward”). MrFrench should mark these tasks as pending each day until the deadline or until the Parent deletes them. He can send daily reminders about today’s recurring tasks.

22. The Parent can delete a task by telling Timmy (“You don’t need to do X anymore”) or telling MrFrench (“Timmy doesn’t have to do this anymore”). MrFrench should detect and remove the task from the database.

23. As before, the conversation flow must be smooth: MrFrench should be friendly with Timmy and professional with the Parent. Parent‑facing responses should feel like natural messages—not emails—and avoid bullet points except when appropriate.

24. Provide an endpoint to manage Timmy Zone.

25. Provide an endpoint to fetch tasks filtered by `is_completed` status (“Pending,” “Progress,” “Completed”) for display in the frontend’s sidebar.

**Important Note:** All examples above are illustrative. MrFrench must understand varied phrasing and extract the intended meaning, not rely on exact wording.