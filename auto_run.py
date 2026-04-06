# auto_run.py

from inference import run_episode
from tasks import grade_easy, grade_medium, grade_hard


def run_full_evaluation():
    print("🚀 Starting Auto Evaluation...\n")

    state, total_reward = run_episode()

    print("\n🎯 Total Reward:", total_reward)

    # pass STATE (not env)
    easy_score = grade_easy(state)
    medium_score = grade_medium(state)
    hard_score = grade_hard(state)

    print("\n--- TASK SCORES ---")
    print(f"Easy   : {easy_score:.2f}")
    print(f"Medium : {medium_score:.2f}")
    print(f"Hard   : {hard_score:.2f}")

    tickets = state["tickets"]
    total_tickets = len(tickets)
    resolved = sum(1 for t in tickets if t["resolved"])

    print(f"\n✅ Resolved Tickets: {resolved}/{total_tickets}")

    print("\n🎉 Pre-submission check completed successfully!")


if __name__ == "__main__":
    run_full_evaluation()
    