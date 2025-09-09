.PHONY: init lint test run evals serve docker-build docker-run docker-run-ollama

init:
	uv venv || true
	. .venv/bin/activate && uv pip install -e .

run:
	micro-agent ask --question "What's 2*(3+5)? Also, give UTC time."

serve:
	uvicorn micro_agent.server:app --reload --port 8000

evals:
	python evals/run_evals.py --n 30

test:
	pytest -q

docker-build:
	docker build -t dspy-micro-agent:latest .

docker-run:
	# Requires OPENAI_API_KEY in environment
	docker run --rm -p 8000:8000 \
	 -e OPENAI_API_KEY=$$OPENAI_API_KEY \
	 -e OPENAI_MODEL=$${OPENAI_MODEL:-gpt-4o-mini} \
	 -e TRACES_DIR=/data/traces \
	 -v $$(pwd)/traces:/data/traces \
	 dspy-micro-agent:latest

docker-run-ollama:
	# Connects to a host Ollama daemon
	docker run --rm -p 8000:8000 \
	 -e LLM_PROVIDER=ollama \
	 -e OLLAMA_HOST=http://host.docker.internal:11434 \
	 -e OLLAMA_MODEL=$${OLLAMA_MODEL:-llama3.1:8b} \
	 -e TRACES_DIR=/data/traces \
	 -v $$(pwd)/traces:/data/traces \
	 dspy-micro-agent:latest
