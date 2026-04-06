# tasks.py

# -----------------------------
# EASY TASK
# -----------------------------
def grade_easy(state) -> float:
    tickets = state["tickets"]   # ✅ FIX
    resolved = sum(1 for t in tickets if t["resolved"])
    return min(1.0, resolved / 2)


# -----------------------------
# MEDIUM TASK
# -----------------------------
def grade_medium(state) -> float:
    tickets = state["tickets"]   # ✅ FIX

    medium_tickets = [t for t in tickets if t["priority"] == "medium"]

    if not medium_tickets:
        return 1.0

    resolved = sum(1 for t in medium_tickets if t["resolved"])
    return resolved / len(medium_tickets)


# -----------------------------
# HARD TASK
# -----------------------------
def grade_hard(state) -> float:
    tickets = state["tickets"]   # ✅ FIX

    total = len(tickets)
    if total == 0:
        return 1.0

    resolved = sum(1 for t in tickets if t["resolved"])
    resolved_score = resolved / total

    sla_penalty = sum(
        1 for t in tickets if t["waiting_time"] > t["sla_deadline"]
    )
    sla_score = 1 - (sla_penalty / total)

    return 0.7 * resolved_score + 0.3 * sla_score