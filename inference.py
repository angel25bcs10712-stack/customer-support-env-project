import os
import sys
from typing import List
import requests
from pydantic import BaseModel

# --- PATH FIX: Ensures Python can see 'env' and 'server' folders ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add server and env to sys.path explicitly
for folder in ["server", "env"]:
    folder_path = os.path.join(current_dir, folder)
    if os.path.exists(folder_path):
        sys.path.append(folder_path)

# --- IMPORT MODELS: Dynamic fallback for 'server/models.py' ---
try:
    from server.models import Action
except (ImportError, ModuleNotFoundError):
    try:
        from models import Action
    except (ImportError, ModuleNotFoundError):
        # Final safety fallback if the import system fails
        class Action(BaseModel):
            response: str

from openai import OpenAI
from env.environment import CustomerSupportEnv

# Configuration from Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is required.", flush=True)
    sys.exit(1)

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL) if API_BASE_URL else OpenAI(api_key=OPENAI_API_KEY)

SUCCESS_THRESHOLD = 0.7

# --- LOGGING HELPERS: Strict Meta OpenEnv Format ---
def log_start(task: str, model: str):
    print(f"[START] task={task} env=customer-support model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool):
    # Validator requires lowercase 'true/false'
    done_str = str(done).lower()
    print(f"[STEP] step={step} action={action!r} reward={reward:.3f} done={done_str} error=none", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    success_str = str(success).lower()
    print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards}", flush=True)

# --- PROMPT LOGIC ---
def build_prompt(observation) -> str:
    # Handles both Pydantic objects and dictionaries
    step = getattr(observation, 'step', observation.get('step', 1))
    ticket = getattr(observation, 'ticket', observation.get('ticket', ""))
    
    if step == 1:
        return (
            "You are a support assistant. Select one category: delivery, refund, account. "
            f"Reply with ONLY the word.\n\nTicket: {ticket}"
        )
    return f"Write a short, empathetic resolution for this ticket: {ticket}"

def get_llm_response(prompt: str) -> str:
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# --- EVALUATION LOOP ---
def run_task(env: CustomerSupportEnv):
    # 1. Reset (Returns Tuple: obs_dict, info_dict)
    obs_dict, info = env.reset()
    task_name = info.get("task", "unknown")
    log_start(task_name, MODEL_NAME)

    rewards = []
    done = False
    current_obs = obs_dict

    for step in range(1, env.MAX_STEPS + 1):
        if done: break

        prompt = build_prompt(current_obs)
        action_text = get_llm_response(prompt)

        # 2. Step (Returns Tuple: obs_dict, reward, done, info)
        next_obs, reward, done, info = env.step(Action(response=action_text))
        
        rewards.append(float(reward))
        current_obs = next_obs
        log_step(step, action_text, reward, done)

    # 3. Final Scoring
    total_score = sum(rewards)
    success = total_score >= SUCCESS_THRESHOLD
    log_end(success, len(rewards), total_score, rewards)
    return total_score

def main():
    env = CustomerSupportEnv()
    # Run through the task list 3 times
    for _ in range(3):
        run_task(env)

if __name__ == "__main__":
    main()
       
    
 
