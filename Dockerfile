# Gateway image (portable fallback for Render / any container host). Modal and Vercel use their own builders.
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md* ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" "sqlalchemy>=2" "psycopg[binary]>=3.1" \
    "pydantic>=2.7" "pydantic-settings>=2.3" "pyyaml" "httpx" "numpy"
ENV PYTHONPATH=/app/src BREACHLAB_CONFIG_DIR=/app/configs PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn breachlab.gateway.asgi:app --host 0.0.0.0 --port ${PORT}"]
