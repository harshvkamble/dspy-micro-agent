FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRACES_DIR=/data/traces

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY micro_agent /app/micro_agent
COPY evals /app/evals

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -e . && \
    pip install --no-cache-dir uvicorn

RUN mkdir -p /data/traces

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status==200 else 1)"]

CMD ["uvicorn", "micro_agent.server:app", "--host", "0.0.0.0", "--port", "8000"]
