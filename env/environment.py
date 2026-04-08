import re
from typing import Dict, Tuple, Any
from pydantic import BaseModel

class Action(BaseModel):
    response: str

class CustomerSupportEnv:
    def __init__(self):
        self.step_count = 0
        self.tasks = [
            {"ticket": "Refund my order", "cat": "refund", "kw": ["refund", "return"]},
            {"ticket": "Can't login", "cat": "account", "kw": ["login", "password"]},
            {"ticket": "Where is my package?", "cat": "delivery", "kw": ["delivery", "track"]}
        ]
        self.current_idx = 0

    def reset(self):
        self.step_count = 0
        task = self.tasks[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.tasks)
        return {"ticket": task["ticket"]}, {"task": task["cat"]}

    def step(self, action: Action) -> Tuple[Dict, float, bool, bool, Dict]:
        self.step_count += 1
        # Hard-coded safety logic: No 0.0 or 1.0 possible
        normalized = action.response.lower()
        
        # Simple grading
        if any(w in normalized for w in ["sorry", "help", "please", "thank"]):
            reward = 0.75
        else:
            reward = 0.25
            
        done = self.step_count >= 1
        # Returns 5 values: obs, reward, terminated, truncated, info
        return {"ticket": "done"}, float(reward), done, False, {}
