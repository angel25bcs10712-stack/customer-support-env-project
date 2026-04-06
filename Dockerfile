# -----------------------------
# Dockerfile for Customer Support RL Environment
# -----------------------------

# Use official Python image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Set environment variable to prevent Python from buffering stdout
ENV PYTHONUNBUFFERED=1

# Default command to run the inference script
CMD ["python", "inference.py"]