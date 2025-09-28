FROM python:3.11-slim

WORKDIR /app

# Copy enhanced backend API
COPY enhanced_dashboard_server.py /app/

EXPOSE 8080

CMD ["python", "/app/enhanced_dashboard_server.py", "8080"]
