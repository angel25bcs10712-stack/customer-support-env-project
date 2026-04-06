from typing import Dict
import random
from models import EnvObservation, EnvAction, EnvReward

class CustomerSupportEnv:
    
    def __init__(self):
        self.tickets_data: Dict = {}   # renamed
        self.time: int = 0

    def reset(self) -> EnvObservation:
        self.time = 0
        self.tickets_data = {"tickets": []}
        
        for _ in range(3):
            self._add_ticket()
            
        return EnvObservation(
            tickets=self.tickets_data["tickets"],
            time=self.time
        )

    def _add_ticket(self):
        ticket = {
            "id": len(self.tickets_data["tickets"]),
            "priority": random.choice(["low", "medium", "high"]),
            "resolved": False,
            "waiting_time": 0,
            "sla_deadline": random.randint(3, 7)
        }
        self.tickets_data["tickets"].append(ticket)

    def step(self, action: EnvAction):
        self.time += 1
        reward = 0.0
        info = {}

        if action.ticket_id >= len(self.tickets_data["tickets"]):
            reward -= 0.5
        else:
            ticket = self.tickets_data["tickets"][action.ticket_id]

            if ticket["resolved"]:
                reward -= 0.2
            else:
                ticket["resolved"] = True

                if ticket["priority"] == "high":
                    reward += 1.0
                elif ticket["priority"] == "medium":
                    reward += 0.7
                else:
                    reward += 0.5

        for t in self.tickets_data["tickets"]:
            if not t["resolved"]:
                t["waiting_time"] += 1
                if t["waiting_time"] > t["sla_deadline"]:
                    reward -= 0.3

        if random.random() < 0.3:
            self._add_ticket()

        done = all(t["resolved"] for t in self.tickets_data["tickets"]) or self.time >= 20

        obs = EnvObservation(
            tickets=self.tickets_data["tickets"],
            time=self.time
        )

        r = EnvReward(value=reward)

        return obs, r, done, info

    def get_state(self) -> EnvObservation:
        return EnvObservation(
            tickets=self.tickets_data["tickets"],
            time=self.time
        )