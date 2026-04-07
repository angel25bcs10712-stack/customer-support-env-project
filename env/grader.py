import re
from typing import Dict

CATEGORY_SYNONYMS = {
    "delivery": ["delivery", "shipping", "shipment", "track", "tracked", "arrive"],
    "refund": ["refund", "return", "reimburse", "money back", "exchange"],
    "account": ["account", "login", "access", "locked", "sign in", "billing", "subscription"],
}

EMPATHY_PHRASES = [
    "sorry",
    "apologize",
    "understand",
    "thank you",
    "happy to help",
    "please",
]

NEGATIVE_PHRASES = [
    "can't",
    "cannot",
    "won't",
    "not possible",
    "no",
    "unable",
]

URGENT_PHRASES = [
    "urgent",
    "as soon as possible",
    "right away",
    "immediately",
    "priority",
]


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def contains_any(text: str, keywords) -> bool:
    return any(keyword in text for keyword in keywords)


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
    keyword_matches = [keyword for keyword in task["resolution_keywords"] if keyword in normalized]
    coverage = len(keyword_matches) / max(len(task["resolution_keywords"]), 1)
    score = 0.2 + coverage * 0.4

    if task["expected_category"] == "account" and contains_any(normalized, ["unlock", "access", "billing", "escalate"]):
        score += 0.05
    if task["expected_category"] == "refund" and contains_any(normalized, ["refund", "return", "reimburse"]):
        score += 0.05
    if task["expected_category"] == "delivery" and contains_any(normalized, ["tracking", "shipment", "shipping"]):
        score += 0.05

    if contains_any(normalized, NEGATIVE_PHRASES):
        score -= 0.1

    return max(min(score, 0.7), 0.0)


def grade_classification(response: str, task: Dict[str, str]) -> float:
    category_match = best_category_match(response, task["expected_category"])
    return round(category_match * 0.3, 3)


def grade_resolution(response: str, task: Dict[str, str]) -> float:
    return round(resolution_score(response, task) + empathy_score(response, task["sentiment"]), 3)


def grade(response: str, task: Dict[str, str], step: int) -> float:
    if step == 1:
        return grade_classification(response, task)
    return grade_resolution(response, task)
