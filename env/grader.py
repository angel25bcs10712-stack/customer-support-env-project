import re
from typing import Dict

# --- 1. CONFIGURATION DATA ---
CATEGORY_SYNONYMS = {
    "delivery": ["delivery", "shipping", "shipment", "track", "tracked", "arrive"],
    "refund": ["refund", "return", "reimburse", "money back", "exchange"],
    "account": ["account", "login", "access", "locked", "sign in", "billing", "subscription"],
}

EMPATHY_PHRASES = ["sorry", "apologize", "understand", "thank you", "happy to help", "please"]
NEGATIVE_PHRASES = ["can't", "cannot", "won't", "not possible", "no", "unable"]
URGENT_PHRASES = ["urgent", "as soon as possible", "right away", "immediately", "priority"]

# --- 2. UTILITY FUNCTIONS ---
def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())

def contains_any(text: str, keywords) -> bool:
    return any(keyword in text for keyword in keywords)

# --- 3. SCORING MODULES ---
def best_category_match(response: str, expected: str) -> float:
    normalized = normalize_text(response)
    if contains_any(normalized, CATEGORY_SYNONYMS[expected]):
        return 1.0
    
    other_categories = [k for k in CATEGORY_SYNONYMS if k != expected]
    if any(contains_any(normalized, CATEGORY_SYNONYMS[c]) for c in other_categories):
        return 0.2
    return 0.0

def empathy_score(response: str, sentiment: str) -> float:
    normalized = normalize_text(response)
    score = 0.0
    if contains_any(normalized, EMPATHY_PHRASES):
        score += 0.15
    if sentiment == "angry" and contains_any(normalized, URGENT_PHRASES):
        score += 0.05
    return min(score, 0.2)

def resolution_score(response: str, task: Dict[str, str]) -> float:
    normalized = normalize_text(response)
    keywords = task.get("resolution_keywords", [])
    
    keyword_matches = [kw for kw in keywords if kw in normalized]
    coverage = len(keyword_matches) / max(len(keywords), 1)
    
    score = 0.2 + (coverage * 0.4)
    
    cat = task.get("expected_category")
    if cat == "account" and contains_any(normalized, ["unlock", "access", "billing", "escalate"]):
        score += 0.05
    if cat == "refund" and contains_any(normalized, ["refund", "return", "reimburse"]):
        score += 0.05
    if cat == "delivery" and contains_any(normalized, ["tracking", "shipment", "shipping"]):
        score += 0.05
        
    if contains_any(normalized, NEGATIVE_PHRASES):
        score -= 0.1
        
    return max(min(score, 0.7), 0.0)

# --- 4. MAIN GRADING INTERFACE ---
def grade_classification(response: str, task: Dict[str, str]) -> float:
    category_match = best_category_match(response, task["expected_category"])
    return round(category_match * 0.3, 3)

def grade_resolution(response: str, task: Dict[str, str]) -> float:
    raw_res = resolution_score(response, task)
    raw_emp = empathy_score(response, task.get("sentiment", "neutral"))
    return round(raw_res + raw_emp, 3)

import re
from typing import Dict

# ... (Keep your CATEGORY_SYNONYMS and helper functions as they are) ...

def grade(response: str, task: Dict[str, str], step: int) -> float:
    """
    Main entry point for the environment grader.
    """
    if step == 1:
        # grade_classification logic
        category_match = best_category_match(response, task["expected_category"])
        raw_score = round(category_match * 0.3, 3)
    else:
        # grade_resolution logic
        raw_res = resolution_score(response, task)
        raw_emp = empathy_score(response, task.get("sentiment", "neutral"))
        raw_score = round(raw_res + raw_emp, 3)
    
    # --- CRITICAL VALIDATOR FIX ---
    # The validator requires scores STRICTLY between 0 and 1.
    # This 'clamps' the score to [0.01, 0.99].
    safe_score = max(0.01, min(raw_score, 0.99))
    
    return float(round(safe_score, 3))
