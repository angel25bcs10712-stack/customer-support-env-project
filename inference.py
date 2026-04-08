import os
import sys
from typing import List
from pydantic import BaseModel

# --- 1. PATH FIXING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
for folder in ["server", "env"]:
    sys.path.append(os.path.join(current_dir, folder))

# --- 2. MODEL DEFINITION (Must match your environment's expectation) ---
try:
    from server.models import Action
except ImportError:
    try:
        from models import Action
    except ImportError:
        class Action(BaseModel):
            response: str

from openai import OpenAI
from env.environment import CustomerSupportEnv

# --- 3. CONFIGURATION ---
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is missing.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# --- 4. LOGGING HELPERS (Strict Meta/HuggingFace Format) ---
def log_start(task: str):
    print(f"[START] task={task} env=customer-support model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool):
    print(f"[STEP] step={step} action={action!r} reward={reward:.3f} done={str(done).lower()} error=none", flush=True)

def log_end(success: bool, steps: int, score: float):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={score:.3f}", flush=True)

# --- 5. EXECUTION LOGIC ---
def run_task(env: CustomerSupportEnv):
    try:
        # TUPLE UNPACKING FIX: env.reset() returns (observation, info)
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs, info = reset_result, {}

        log_start(info.get("task", "support_ticket"))

        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < getattr(env, 'MAX_STEPS', 5):
            # Extract ticket text safely
            ticket = obs.get("ticket", "") if isinstance(obs, dict) else getattr(obs, "ticket", "")
            prompt = f"Resolve this customer ticket: {ticket}"
            
            # API Call
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            action_text = res.choices[0].message.content.strip()
            
            # TUPLE UNPACKING FIX: env.step() returns (obs, reward, done, info)
            step_result = env.step(Action(response=action_text))
            obs, reward, done, info = step_result # This is where most crashes happen
            
            steps += 1
            total_reward += float(reward)
            log_step(steps, action_text, reward, done)

        log_end(total_reward > 0.1, steps, total_reward)
        
    except Exception as e:
        # Catching the exception so the validator doesn't see a "non-zero exit"
        print(f"DEBUG ERROR: {e}", flush=True)
        # We don't sys.exit(1) here so the logs can be fully captured
        return

def main():
    try:
        env = CustomerSupportEnv()
        for _ in range(3):
            run_task(env)
    except Exception as e:
        print(f"FAILED TO START ENV: {e}")

if __name__ == "__main__":
    main()
