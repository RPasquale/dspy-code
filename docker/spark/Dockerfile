# Simple Spark image with project code at /opt/app
# Usage:
#   docker build -f docker/spark/Dockerfile -t ghcr.io/OWNER/dspy-spark:latest .

FROM bitnami/spark:3.5.1

USER root

# Install minimal Python dependencies for streaming jobs
RUN pip install --no-cache-dir \
    pyarrow==16.1.0 pyspark==3.5.1 \
    python-docx==1.1.2 docx2txt==0.8 pdfminer.six==20231228

# Copy application code
WORKDIR /opt/app
COPY streaming /opt/app/streaming
COPY dspy_agent /opt/app/dspy_agent
COPY scripts/spark_job_supervisor.py /opt/app/scripts/spark_job_supervisor.py

# Optional: set PYTHONPATH so jobs can import streaming modules via local path
ENV PYTHONPATH=/opt/app:${PYTHONPATH}

USER 1001
