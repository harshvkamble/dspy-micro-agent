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
	docker build -t dspy-micro-agent .

docker-run:
	# Pass OPENAI_API_KEY from your environment if using OpenAI
	docker run --rm -p 8000:8000 -e OPENAI_API_KEY dspy-micro-agent

docker-run-ollama:
	# Assumes Ollama running on host at 11434; adjust host mapping if needed
	docker run --rm -p 8000:8000 \
	  -e LLM_PROVIDER=ollama \
	  -e OLLAMA_MODEL=llama3.2:1b \
	  -e OLLAMA_HOST=http://host.docker.internal:11434 \
	  dspy-micro-agent
