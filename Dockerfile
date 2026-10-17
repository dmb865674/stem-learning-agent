# Multi-stage / lightweight production container for Google Cloud Run
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user for security best practices on Cloud Run
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose default Cloud Run port
EXPOSE 8080

# Cloud Run injects $PORT; exec form ensures signals (SIGTERM) are handled properly
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
