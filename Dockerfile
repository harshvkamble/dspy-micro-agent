# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:0.4.17 as builder
ARG PYTHON_VERSION
WORKDIR /app
COPY pyproject.toml README.md .
COPY micro_agent micro_agent
COPY evals evals
COPY tests tests

ENV UV_HTTP_TIMEOUT=60
RUN uv venv --python $PYTHON_VERSION && . .venv/bin/activate && uv pip install -e .

FROM python:${PYTHON_VERSION}-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_HTTP_TIMEOUT=60 \
    TRACES_DIR=/data/traces

RUN useradd -m appuser
WORKDIR /app
COPY --from=builder /app /.build
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source for runtime (editable install already baked in)
COPY micro_agent micro_agent
COPY evals evals
COPY README.md README.md

RUN mkdir -p /data/traces && chown -R appuser:appuser /data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python - <<'PY' || exit 1
import json, sys, urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:8000/openapi.json', timeout=3) as r:
        sys.exit(0 if r.status==200 else 1)
except Exception:
    sys.exit(1)
PY

CMD ["uvicorn", "micro_agent.server:app", "--host", "0.0.0.0", "--port", "8000"]

