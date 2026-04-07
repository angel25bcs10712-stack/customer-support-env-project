from pydantic import BaseModel


class Observation(BaseModel):
    ticket: str
    priority: str
    sentiment: str
    sla_hours: float
    step: int
    instructions: str


class Action(BaseModel):
    response: str


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict
