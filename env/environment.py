import yaml
from typing import Dict, Any, Tuple
# Import the central models from the server package
from server.models import Observation, StepResponse, Action

# Alias StepResponse to StepResult so the existing method signatures work
StepResult = StepResponse

# Import logic from sibling files in the same folder
from .tasks import TASKS
from .grader import grade

class CustomerSupportEnv:
    MAX_STEPS = 2

    def __init__(self):
        self.index = 0
        self.current_task = None
        self.step_number = 0
        self.total_reward = 0.0
        self.done = False

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Resets the environment for a new task.
        Standard OpenEnv reset returns (observation_dict, info_dict).
        """
        self.current_task = TASKS[self.index % len(TASKS)]
        self.index += 1
        self.step_number = 1
        self.total_reward = 0.0
        self.done = False

        obs = self._build_observation()
        info = {
            "task": self.current_task.get("name", "Support Task"),
            "difficulty": self.current_task.get("difficulty", "medium"),
            "phase": "classification",
        }
        # Returning as dicts is the most compatible format for the FastAPI layer
        return obs.dict(), info

    def step(self, action_input: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Processes an agent action and returns (observation, reward, done, info).
        Handles both raw strings and Action objects for maximum compatibility.
        """
        if self.current_task is None:
            raise ValueError("Environment must be reset before calling step().")
        
        # Extract the response string regardless of how the API sent it
        if hasattr(action_input, 'response'):
            response_text = action_input.response
        elif isinstance(action_input, dict):
            response_text = action_input.get("response", "")
        else:
            response_text = str(action_input)

        current_step = self.step_number
        
        # Calculate reward using your custom grader
        reward = grade(response_text, self.current_task, current_step)
        self.total_reward += reward
        
        # Determine if the session is finished
        self.done = (current_step >= self.MAX_STEPS)
        
        if not self.done:
            self.step_number += 1

        obs = self._build_observation()
        info = {
            "task": self.current_task.get("name", "Support Task"),
            "difficulty": self.current_task.get("difficulty", "medium"),
            "phase": "resolution" if self.done else "classification",
        }

        return obs.dict(), reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """Returns current environment metadata."""
        return {
            "task": self.current_task["name"] if self.current_task else None,
            "step": self.step_number,
            "total_reward": self.total_reward,
            "is_done": self.done
        }

    def close(self) -> None:
        """Cleanup logic if necessary."""
        pass

    def _build_observation(self) -> Observation:
        """Helper to construct the Observation Pydantic model."""
        return Observation(
            ticket=self.current_task["ticket"],
            priority=self.current_task["priority"],
            sentiment=self.current_task["sentiment"],
            sla_hours=self.current_task["sla_hours"],
            step=self.step_number,
            instructions=self.current_task["instructions"],
        )
