# api.py

from fastapi import FastAPI
from env import CustomerSupportEnv
from models import Action

app = FastAPI(title="Customer Support API")

# Initialize environment
env = CustomerSupportEnv()


# -----------------------------
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"message": "Customer Support API is running"}


# -----------------------------
# RESET
# -----------------------------
@app.post("/reset")
def reset():
    state = env.reset()
    return {
        "state": state.dict(),
        "message": "Environment reset"
    }


# -----------------------------
# STEP
# -----------------------------
@app.post("/step")
def step(action: Action):
    state, reward, done, info = env.step(action)

    return {
        "state": state.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }