FROM python:3.11-slim

WORKDIR /app

COPY dspy_agent /app/dspy_agent
COPY scripts /app/scripts

RUN pip install --no-cache-dir fastapi uvicorn[standard] fastembed onnxruntime numpy sentence-transformers

ENV MODEL_ID=BAAI/bge-small-en-v1.5
EXPOSE 9000
ENTRYPOINT ["uvicorn", "dspy_agent.embedding.infermesh_server:app", "--host", "0.0.0.0", "--port", "9000"]
