# Spark training image with PyTorch and Transformers
# Build: docker build -f docker/spark-train/Dockerfile -t ghcr.io/OWNER/dspy-spark-train:latest .

FROM bitnami/spark:3.5.1

USER root

# Python deps for training
RUN pip install --no-cache-dir \
    pyarrow==16.1.0 pyspark==3.5.1 \
    torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    transformers==4.43.3 accelerate==0.32.1

# Copy application code
WORKDIR /opt/app
COPY streaming /opt/app/streaming
COPY dspy_agent /opt/app/dspy_agent

ENV PYTHONPATH=/opt/app:${PYTHONPATH}

USER 1001

