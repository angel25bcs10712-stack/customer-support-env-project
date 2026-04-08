import re
from typing import Dict, Any, List
# --- 1. CONFIGURATION ---
# We use fixed scores to ensure we never touch the 0.0 or 1.0 boundaries.
SCORE_MAP = {
    "MATCH_HIGH": 0.850,
    "MATCH_LOW": 0.450,
    "MISS": 0.150
}

CATEGORY_KEYWORDS = {
    "delivery": ["delivery", "shipping", "track", "arrive", "package"],
    "refund": ["refund", "return", "money", "reimburse"],
    "account": ["login", "password", "account", "access", "billing"]
}

# --- 2. UTILITY ---
def clean_text(text: str) -> str:
    """Standardizes text to ensure robust keyword matching."""
    if not text:
        return ""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())

# --- 3. MAIN GRADER ---
def grade(response: str, task: Dict[str, Any], step: int) -> float:
    """
    Final Safety Grader.
    GUARANTEES a return value strictly inside the (0, 1) open interval.
    """
    normalized = clean_text(response)
    expected_cat = task.get("expected_category", "account")
    
    # 1. Logic Selection
    if step == 1:
        # Check for Category match
        keywords = CATEGORY_KEYWORDS.get(expected_cat, ["help"])
        is_match = any(word in normalized for word in keywords)
        raw_score = SCORE_MAP["MATCH_LOW"] if is_match else SCORE_MAP["MISS"]
    else:
        # Check for Resolution keywords
        res_keywords = task.get("resolution_keywords", [])
        is_match = any(word in normalized for word in res_keywords) if res_keywords else False
        raw_score = SCORE_MAP["MATCH_HIGH"] if is_match else SCORE_MAP["MISS"]

    # 2. THE FINAL HARD LOCK
    # This is the 'Shield'. Even if logic above is modified, 
    # this code prevents a 0.0 or 1.0 from ever escaping.
    safe_score = max(0.100, min(float(raw_score), 0.900))
    
    # Returning a float rounded to 3 decimal places
    return float(round(safe_score, 3))
