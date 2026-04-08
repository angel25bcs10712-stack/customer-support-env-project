import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from server.models import StepRequest, StepResponse  # Correct package import
from env.environment import CustomerSupportEnv

app = FastAPI(
    title="Customer Support OpenEnv",
    description="A customer support ticket triage and response environment for agent evaluation.",
)

# Initialize the environment globally within the app
env = CustomerSupportEnv()

@app.get("/")
def root():
    """Redirects users to the interactive API documentation."""
    return RedirectResponse(url="/docs")

@app.post("/reset")
def reset():
    """Resets the environment and returns the initial observation."""
    observation, info = env.reset()
    # Formatting to match the expected OpenEnv StepResponse schema
    return {
        "observation": observation,
        "reward": 0.0,
        "done": False,
        "info": info,
    }

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Executes a single step in the environment."""
    try:
        # Note: We pass the string directly or wrap it if your Env expects an object
        observation, reward, done, info = env.step(request.response)
        
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        # This helps the validator identify if your logic crashes
        raise HTTPException(status_code=400, detail=f"Step failed: {str(e)}")

@app.get("/state")
def state():
    """Returns the current internal state of the environment."""
    return {"state": env.state()}

def main():
    """Entry point for the 'server' command in pyproject.toml."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
