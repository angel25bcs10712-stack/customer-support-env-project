import os
import yaml
import re
from typing import Dict, List, Tuple, Any
from pydantic import BaseModel

# --- 1. MODEL DEFINITION ---
class Action(BaseModel):
    response: str

# --- 2. THE GRADER (Scoring Logic) ---
class SupportGrader:
    CATEGORY_SYNONYMS = {
        "delivery": ["delivery", "shipping", "shipment", "track", "tracked", "arrive"],
        "refund": ["refund", "return", "reimburse", "money back", "exchange"],
        "account": ["account", "login", "access", "locked", "sign in", "billing", "subscription"],
    }
    
    EMPATHY_PHRASES = ["sorry", "apologize", "understand", "thank you", "happy to help", "please"]
    NEGATIVE_PHRASES = ["can't", "cannot", "won't", "not possible", "no", "unable"]

    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", " ", text.lower())

    def grade(self, response: str, task: Dict, step: int) -> float:
        normalized = self.normalize_text(response)
        expected_cat = task.get("expected_category", "account")
        
        # Step 1: Classification Score
        if step == 1:
            synonyms = self.CATEGORY_SYNONYMS.get(expected_cat, [])
            match = any(word in normalized for word in synonyms)
            raw_score = 0.3 if match else 0.05
        
        # Step 2+: Resolution Score
        else:
            keywords = task.get("resolution_keywords", [])
            matches = sum(1 for kw in keywords if kw in normalized)
            coverage = matches / max(len(keywords), 1)
            
            # Base logic + Empathy bonus
            empathy_bonus = 0.15 if any(w in normalized for w in self.EMPATHY_PHRASES) else 0.0
            penalty = 0.1 if any(w in normalized for w in self.NEGATIVE_PHRASES) else 0.0
            raw_score = (coverage * 0.5) + empathy_bonus - penalty

        # --- THE CRITICAL VALIDATOR FIX ---
        # "Strictly between 0 and 1" means no 0.0 and no 1.0.
        # This clamping ensures the Task Validation check passes.
        safe_score = max(0.01, min(raw_score, 0.99))
        return float(round(safe_score, 3))

# --- 3. THE ENVIRONMENT ---
class CustomerSupportEnv:
    def __init__(self):
        self.grader = SupportGrader()
        self.tasks = [
            {"ticket": "Where is my order?", "expected_category": "delivery", "resolution_keywords": ["tracking", "transit"]},
            {"ticket": "I want a refund for my broken item.", "expected_category": "refund", "resolution_keywords": ["return", "refund", "receipt"]},
            {"ticket": "I cannot log into my account.", "expected_category": "account", "resolution_keywords": ["password", "reset", "email"]}
        ]
        self.current_task_idx = 0
        self.step_count = 0
        self.MAX_STEPS = 2

    def reset(self) -> Tuple[Dict, Dict]:
        self.step_count = 0
        task = self.tasks[self.current_task_idx]
        self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
        return {"ticket": task["ticket"]}, {"task": task["expected_category"]}

    def step(self, action: Action) -> Tuple[Dict, float, bool, bool, Dict]:
        self.step_count += 1
        task = self.tasks[(self.current_task_idx - 1) % len(self.tasks)]
        
        # Calculate Reward using the Clamped Grader
        reward = self.grader.grade(action.response, task, self.step_count)
        
        done = self.step_count >= self.MAX_STEPS
        return {"ticket": "Next step..."}, reward, done, False, {}
