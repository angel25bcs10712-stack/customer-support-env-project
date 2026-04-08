from pydantic import BaseModel
from typing import Any, Dict

# This was the missing class causing the ImportError
class Action(BaseModel):
    response: str

class Observation(BaseModel):
    ticket: str
    priority: str
    sentiment: str
    sla_hours: float
    step: int
    instructions: str

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

# Adding this alias ensures your env/environment.py 
# can use 'StepResult' without crashing
StepResult = StepResponse

class StepRequest(BaseModel):
    response: str

class StateResponse(BaseModel):
    step_count: int
    total_reward: float
    is_done: bool
