FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (optional, keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for better Docker caching
COPY pyproject.toml README.md /app/

# Copy source
COPY micro_agent /app/micro_agent
COPY evals /app/evals

# Install package
RUN pip install --upgrade pip && \
    pip install . && \
    pip install uvicorn

EXPOSE 8000

CMD ["uvicorn", "micro_agent.server:app", "--host", "0.0.0.0", "--port", "8000"]

