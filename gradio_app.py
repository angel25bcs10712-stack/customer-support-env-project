import gradio as gr
from env import CustomerSupportEnv

# Action class (same as inference)
class Action:
    def __init__(self, ticket_id):
        self.ticket_id = ticket_id


env = CustomerSupportEnv()
state = env.reset()


def reset_env():
    global env, state
    env = CustomerSupportEnv()
    state = env.reset()
    return {"tickets": state.tickets}


def step_env(action):
    global state
    action_obj = Action(int(action))

    state, reward, done, _ = env.step(action_obj)

    return {
        "tickets": state.tickets,
        "reward": reward.value,
        "done": done
    }


def auto_run_env():
    global env, state
    env = CustomerSupportEnv()
    state = env.reset()

    done = False
    total_reward = 0

    while not done:
        # simple greedy
        tickets = state.tickets
        unresolved = [t for t in tickets if not t["resolved"]]
        if not unresolved:
            break

        action_obj = Action(unresolved[0]["id"])
        state, reward, done, _ = env.step(action_obj)
        total_reward += reward.value

    return {
        "tickets": state.tickets,
        "total_reward": total_reward
    }


with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Customer Support RL Environment")

    gr.Markdown("## 🔄 Reset Environment")
    reset_btn = gr.Button("Reset")
    reset_output = gr.JSON()
    reset_btn.click(fn=reset_env, inputs=[], outputs=reset_output)

    gr.Markdown("## ▶️ Take Step")
    action_input = gr.Number(label="Enter Ticket ID")
    step_btn = gr.Button("Step")
    step_output = gr.JSON()
    step_btn.click(fn=step_env, inputs=action_input, outputs=step_output)

    gr.Markdown("## ⚡ Auto Run")
    auto_btn = gr.Button("Run Auto")
    auto_output = gr.JSON()
    auto_btn.click(fn=auto_run_env, inputs=[], outputs=auto_output)


demo.launch()