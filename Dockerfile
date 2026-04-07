FROM python:3.12-slim

# Set environment variables for Python and logs
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a non-root user for Hugging Face security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Standard Hugging Face port
EXPOSE 7860

# Run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
 
 
 
 
 
 

