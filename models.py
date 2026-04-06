# models.py

from pydantic import BaseModel
from typing import List, Dict, Any


# -----------------------------
# ACTION (API use)
# -----------------------------
class Action(BaseModel):
    ticket_id: int


# -----------------------------
# ENV INTERNAL MODELS
# -----------------------------
class EnvObservation(BaseModel):
    tickets: List[Dict[str, Any]]


class EnvAction(BaseModel):
    ticket_id: int


class EnvReward(BaseModel):
    value: float