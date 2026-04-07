from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.environment import CustomerSupportEnv
from env.models import Action

app = FastAPI(
    title="Customer Support OpenEnv",
    description="A customer support ticket triage and response environment for agent evaluation.",
)

env = CustomerSupportEnv()
current_result = None


class StepRequest(BaseModel):
    response: str


def result_to_dict(result):
    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.post("/reset")
def reset():
    global current_result
    current_result = env.reset()
    return result_to_dict(current_result)


@app.post("/step")
def step(request: StepRequest):
    global current_result
    if current_result is None:
        raise HTTPException(status_code=400, detail="Call /reset before calling /step.")
    action = Action(response=request.response)
    current_result = env.step(action)
    return result_to_dict(current_result)


@app.get("/state")
def state():
    return {"state": env.state()}
 