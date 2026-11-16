# Which questions did the grilling ask about the PRD and the build plan, and what changed?

**Owner:** Build Lead. **When:** Thursday, 45 minutes of grilling and then the edits. **Tool used:** _write it here_. **Full chat:** the file in `ai_logs/` it came from.

Write your own answer before you read the AI's recommended answer. When you accept a recommendation, say why in the last column. At least three rows must show a change to a document.

| Q | Question, as asked | AI's recommended answer | Our answer | Accepted, changed or rejected | What changed, in which document |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

## What did we learn from being grilled?

The most difficult question was whether our system could genuinely make predictions 14 days before hotel arrival. It exposed the limitation that our dataset contains final booking information rather than booking snapshots from earlier dates. We will now clearly distinguish the current MVP demonstration from a genuine production forecasting system, freeze the safety buffer before proof evaluation, and reject any method that violates the client's walk ceiling.
