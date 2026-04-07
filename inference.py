import os
import sys
from typing import List

from openai import OpenAI

from env.environment import CustomerSupportEnv
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable is required.", flush=True)
    sys.exit(1)

if not MODEL_NAME:
    print("ERROR: MODEL_NAME environment variable is required.", flush=True)
    sys.exit(1)

if API_BASE_URL:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=API_BASE_URL,
    )
else:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )

SUCCESS_SCORE_THRESHOLD = 0.7


def log_start(task: str, model_name: str):
    print(f"[START] task={task} env=customer-support model={model_name}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    print(
        f"[STEP] step={step} action={action!r} reward={reward:.3f} done={done} error={error}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(
        f"[END] success={success} steps={steps} score={score:.3f} rewards={rewards}",
        flush=True,
    )


def get_model_response(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        return ""


def build_prompt(observation) -> str:
    if observation.step == 1:
        return (
            "You are a customer support assistant. "
            "Read the customer ticket and select exactly one category from: delivery, refund, account. "
            "Respond with only the category word and no additional commentary.\n\n"
            f"Ticket:\n{observation.ticket}\n\n"
            f"Priority: {observation.priority}\n"
            f"Sentiment: {observation.sentiment}\n"
            f"SLA hours remaining: {observation.sla_hours}\n"
        )

    return (
        "You are a customer support agent. "
        "Write a concise, empathetic resolution message that fixes the customer's issue, "
        "apologizes if needed, and explains the next step clearly.\n\n"
        f"Ticket:\n{observation.ticket}\n\n"
        f"Priority: {observation.priority}\n"
        f"Sentiment: {observation.sentiment}\n"
        f"SLA hours remaining: {observation.sla_hours}\n"
        "Do not repeat the category as the only answer."
    )


def run_task(env: CustomerSupportEnv) -> float:
    result = env.reset()
    task_name = result.info.get("task", "unknown")
    log_start(task_name, MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0

    for step in range(1, env.MAX_STEPS + 1):
        if result.done:
            break

        prompt = build_prompt(result.observation)
        action_text = get_model_response(prompt)

        result = env.step(Action(response=action_text))
        reward = float(result.reward)
        rewards.append(reward)
        steps_taken = step

        log_step(step, action_text, reward, result.done)

        if result.done:
            break

    score = sum(rewards)
    score = max(0.0, min(score, 1.0))
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success, steps_taken, score, rewards)

    return score


def main():
    env = CustomerSupportEnv()
    total_scores: List[float] = []

    for _ in range(3):
        score = run_task(env)
        total_scores.append(score)

    average_score = sum(total_scores) / max(len(total_scores), 1)
    print(f"Overall average score: {average_score:.3f}", flush=True)


if __name__ == "__main__":
    main()
