FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy test suite service
COPY scripts/test_suite_service.py /app/test_suite_service.py

# Copy test files
COPY tests/ /app/tests/
COPY frontend/ /app/frontend/

# Make executable
RUN chmod +x /app/test_suite_service.py

# Set entrypoint
ENTRYPOINT ["python", "/app/test_suite_service.py"]
