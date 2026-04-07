from typing import Dict

from .models import Observation, Action, StepResult
from .tasks import TASKS
from .grader import grade


class CustomerSupportEnv:
    MAX_STEPS = 2

    def __init__(self):
        self.index = 0
        self.current_task = None
        self.step_number = 0

    def reset(self) -> StepResult:
        self.current_task = TASKS[self.index % len(TASKS)]
        self.index += 1
        self.step_number = 1

        return StepResult(
            observation=self._build_observation(),
            reward=0.0,
            done=False,
            info={
                "task": self.current_task["name"],
                "difficulty": self.current_task["difficulty"],
                "phase": "classification",
            },
        )

    def step(self, action: Action) -> StepResult:
        if self.current_task is None:
            raise ValueError("Environment must be reset before calling step().")
        if self.step_number > self.MAX_STEPS:
            raise ValueError("Episode already finished. Call reset() to start a new task.")

        current_step = self.step_number
        reward = grade(action.response, self.current_task, current_step)
        done = current_step == self.MAX_STEPS
        self.step_number = min(self.step_number + 1, self.MAX_STEPS)
        self.step_number = 2 if not done else self.step_number

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            done=done,
            info={
                "task": self.current_task["name"],
                "difficulty": self.current_task["difficulty"],
                "phase": "resolution" if current_step >= 1 else "classification",
            },
        )

    def state(self) -> Dict[str, str]:
        return {
            "task": self.current_task["name"] if self.current_task else None,
            "step": self.step_number,
            "difficulty": self.current_task["difficulty"] if self.current_task else None,
        }

    def close(self) -> None:
        pass

    def _build_observation(self) -> Observation:
        return Observation(
            ticket=self.current_task["ticket"],
            priority=self.current_task["priority"],
            sentiment=self.current_task["sentiment"],
            sla_hours=self.current_task["sla_hours"],
            step=self.step_number,
            instructions=self.current_task["instructions"],
        )
