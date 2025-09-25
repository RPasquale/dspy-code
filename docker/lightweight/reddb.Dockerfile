FROM python:3.11-slim

# Install required packages
RUN pip install --no-cache-dir fastapi uvicorn

# Create app directory
WORKDIR /app

# Copy RedDB server
COPY reddb_server.py /app/reddb_server.py

# Create data directory
RUN mkdir -p /data && chmod 755 /data

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the server
CMD ["python", "/app/reddb_server.py"]
