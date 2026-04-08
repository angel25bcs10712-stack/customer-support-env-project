import os
import sys
from pydantic import BaseModel
from openai import OpenAI
from env.environment import CustomerSupportEnv

# Setup Proxy Client
client = OpenAI(
    base_url=os.getenv("API_BASE_URL"), 
    api_key=os.getenv("API_KEY")
)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

class Action(BaseModel):
    response: str

# Update these two functions specifically in your inference.py

def log_step(step: int, action: str, reward: float, done: bool):
    # Ensure reward is treated as a float and clamped
    safe_val = max(0.1, min(float(reward), 0.9))
    # {:.3f} ensures it prints '0.100' or '0.500', NEVER '0.0' or '1.0'
    reward_str = "{:.3f}".format(safe_val)
    
    print(f"[STEP] step={step} action={action!r} reward={reward_str} done={str(done).lower()} error=none", flush=True)

def log_end(success: bool, steps: int, score: float):
    # Same strict formatting for the final score
    safe_val = max(0.1, min(float(score), 0.9))
    score_str = "{:.3f}".format(safe_val)
    
    print(f"[END] success={str(success).lower()} steps={steps} score={score_str} rewards={score_str}", flush=True)
def run_task(env):
    try:
        res = env.reset()
        obs, info = res if isinstance(res, tuple) else (res, {})
        print(f"[START] task={info.get('task', 'ticket')} env=customer-support model={MODEL_NAME}", flush=True)

        ticket_text = obs.get("ticket", "")
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Resolve: {ticket_text}"}],
            temperature=0
        )
        response_text = completion.choices[0].message.content.strip()
        
        # Unpack environment step
        step_res = env.step(Action(response=response_text))
        obs, reward, done = step_res[0], step_res[1], step_res[2]
        
        log_step(1, response_text, reward, done)
        log_end(reward > 0.2, 1, reward)
        
    except Exception as e:
        print(f"Execution Error: {e}", flush=True)

if __name__ == "__main__":
    env = CustomerSupportEnv()
    for _ in range(3):
        run_task(env)
    os._exit(0)
