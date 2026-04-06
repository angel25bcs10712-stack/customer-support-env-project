# inference.py

import os
import requests

# -----------------------------
# ENV VARIABLES
# -----------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "customer-support-agent")
HF_TOKEN = os.getenv("HF_TOKEN")


# -----------------------------
# ACTION LOGIC
# -----------------------------
def choose_action(state):
    tickets = state["tickets"]

    priority_map = {"high": 3, "medium": 2, "low": 1}

    tickets = sorted(
        tickets,
        key=lambda t: (
            priority_map[t["priority"]],
            t["sla_deadline"] - t["waiting_time"]
        ),
        reverse=True
    )

    for t in tickets:
        if not t["resolved"]:
            return t["id"]

    return 0


# -----------------------------
# RUN EPISODE
# -----------------------------
def run_episode():
    print("START")

    # RESET ENV VIA API
    response = requests.post(f"{API_BASE_URL}/reset")
    state = response.json()["state"]

    done = False
    step = 0
    total_reward = 0

    max_steps = 50  # 🔥 safety limit

    while not done and step < max_steps:
        action = choose_action(state)

        response = requests.post(
            f"{API_BASE_URL}/step",
            json={"ticket_id": action}
        )

        data = response.json()

        state = data["state"]

        # ✅ handle reward safely
        reward_data = data["reward"]
        if isinstance(reward_data, dict):
            reward = reward_data.get("value", 0)
        else:
            reward = reward_data

        done = data["done"]

        total_reward += reward

        print(f"STEP {step} | Action: {action} | Reward: {reward}")
        step += 1

    print("END")

    return state, total_reward