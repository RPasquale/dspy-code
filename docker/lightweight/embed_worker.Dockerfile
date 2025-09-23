FROM python:3.11-slim

WORKDIR /app

COPY docker/lightweight/scripts/embed_worker.py /app/scripts/embed_worker.py

RUN pip install --no-cache-dir kafka-python requests pyarrow fastembed

CMD ["python", "/app/scripts/embed_worker.py"]
