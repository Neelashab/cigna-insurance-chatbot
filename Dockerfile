# Minimal Dockerfile for Cloud Run
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 8080 as Cloud Run services listen on this port by default
ENV PORT 8080
EXPOSE 8080

# Run the application when the container launches
CMD ["python3", "api.py"]