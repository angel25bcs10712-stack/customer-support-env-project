# 🚀 Customer Support RL Environment

## 📌 Overview

This project simulates a **Customer Support Reinforcement Learning Environment** where an agent prioritizes and resolves support tickets based on urgency, SLA deadlines, and priority levels.

The system is built using:

* **FastAPI** for backend API
* **Custom RL Environment**
* **Greedy Agent Strategy**
* **Gradio UI** for interaction

---

## ⚙️ Features

* 📡 API-based environment (`/reset`, `/step`)
* 🤖 Intelligent agent for ticket prioritization
* 📊 Evaluation metrics (Easy, Medium, Hard tasks)
* 🖥️ Interactive UI using Gradio
* 🔁 Automated episode execution

---

## 🗂️ Project Structure

```
openenv-project/
│
├── api.py
├── env.py
├── models.py
├── inference.py
├── auto_run.py
├── tasks.py
├── gradio_app.py
│
├── requirements.txt
├── README.md
```

---

## 🔧 Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### 🔹 Start API Server

```bash
uvicorn api:app --reload
```

---

### 🔹 Run Auto Evaluation

```bash
python auto_run.py
```

### ✅ Expected Output Format

```
START
STEP 0 | Action: X | Reward: Y
STEP 1 | Action: X | Reward: Y
...
END
```

---

### 🔹 Run Gradio UI

```bash
python gradio_app.py
```

---

## 🔌 API Endpoints

### 🔹 Reset Environment

```
POST /reset
```

### 🔹 Take Step

```
POST /step
Body:
{
  "ticket_id": int
}
```

---

## 🤖 Agent Logic

The agent selects actions using a **priority-based heuristic**:

* High priority > Medium > Low
* Considers SLA urgency:

  ```
  urgency = sla_deadline - waiting_time
  ```
* Picks the most critical unresolved ticket

---

## 📊 Evaluation Metrics

### ✅ Easy

* Resolve at least 2 tickets

### ✅ Medium

* Resolve all medium-priority tickets

### ✅ Hard

* Maximize:

  * Total resolved tickets
  * Minimize SLA violations

---

## 🌍 Environment Variables

```python
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "customer-support-agent")
HF_TOKEN = os.getenv("HF_TOKEN")
```

---

## 🚀 Submission Notes

* Uses **API-based architecture** (as required)
* Follows **structured logging format**
* No direct environment calls in agent
* Fully compatible with Hugging Face deployment

---

## 🎉 Final Status

✅ API Working
✅ Agent Working
✅ Evaluation Working
✅ UI Working
✅ Ready for Submission

---
