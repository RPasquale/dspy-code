FROM python:3.11-slim

WORKDIR /app

COPY dspy_agent /app/dspy_agent
COPY scripts /app/scripts

RUN pip install --no-cache-dir kafka-python requests pyarrow fastembed pandas

CMD ["python", "-m", "dspy_agent.embedding.embed_worker"]
