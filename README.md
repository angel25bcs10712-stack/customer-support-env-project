# Customer Support OpenEnv Environment

## Overview

This repository implements an **OpenEnv-compatible AI customer support environment**.
The environment simulates ticket triage, issue categorization, and customer-facing resolution messaging with measurable rewards.
It is designed for real-world agent evaluation and includes an API server, baseline inference script, Docker container, and OpenEnv metadata.

## Environment Description

The task is a customer support workflow where an AI agent:
1. Classifies the ticket category.
2. Writes a customer-facing resolution message.

The environment delivers partial feedback after each step and returns a final score from `0.0` to `1.0`.

## Observation Space

Each observation includes:

* `ticket`: the customer's message
* `priority`: support urgency (`low`, `medium`, `high`)
* `sentiment`: customer emotional tone
* `sla_hours`: remaining SLA window in hours
* `step`: current action step number
* `instructions`: guidance for the current phase

## Action Space

* `response`: a single support agent response string

The agent uses the same action type for both classification and resolution steps.

## Tasks and Difficulty Progression

The environment includes three deterministic tasks:

* `easy`: delivery update with clear tracking language
* `medium`: damaged product refund and return assistance
* `hard`: duplicate billing, locked account, and urgent access restoration

Each task is scored by a grader that rewards correct category choice, supportive tone, and a relevant resolution plan.

## OpenEnv Metadata

* `openenv.yaml` defines the entrypoint, task list, observations, actions, and reward type.
* `env.environment:CustomerSupportEnv` is the OpenEnv entrypoint.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API Server

```bash
uvicorn api:app --reload
```

Then call:

* `POST /reset`
* `POST /step` with `{ "response": "..." }`

## Baseline Inference

Set environment variables first:

```bash
set OPENAI_API_KEY=your_api_key
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
```

Run:

```bash
python inference.py
```

The script emits structured logs in the required format:

* `[START] task=... env=... model=...`
* `[STEP] step=... action=... reward=... done=... error=...`
* `[END] success=... steps=... score=... rewards=...`

## Docker

Build and run the container:

```bash
docker build -t customer-support-openenv .
docker run --rm -p 7860:7860 -e OPENAI_API_KEY="$OPENAI_API_KEY" customer-support-openenv
```

## Project Structure

```
openenv-project/
+-- api.py
+-- Dockerfile
+-- README.md
+-- inference.py
+-- openenv.yaml
+-- requirements.txt
+-- env/
¦   +-- __init__.py
¦   +-- environment.py
¦   +-- grader.py
¦   +-- models.py
¦   +-- tasks.py
```

## Notes

* The environment supports multi-step evaluation with partial reward shaping.
* The grader is deterministic and returns values in the `0.0`–`1.0` range.
* The API server is ready for containerized Hugging Face deployment.
