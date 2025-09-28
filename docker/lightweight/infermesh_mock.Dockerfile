FROM python:3.11-slim
WORKDIR /app
COPY dspy_agent /app/dspy_agent
COPY scripts /app/scripts
COPY entrypoints/run_infermesh_mock.sh /entrypoints/run_infermesh_mock.sh
RUN chmod +x /entrypoints/run_infermesh_mock.sh
EXPOSE 9000
ENTRYPOINT ["/entrypoints/run_infermesh_mock.sh"]
