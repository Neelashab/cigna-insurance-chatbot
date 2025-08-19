# Minimal Dockerfile for Cloud Run
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the application
CMD exec uvicorn api:app --host 0.0.0.0 --port $PORT