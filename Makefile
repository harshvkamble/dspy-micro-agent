.PHONY: init lint test run evals serve

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

