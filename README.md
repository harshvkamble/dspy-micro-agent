# DSPy Micro Agent

Minimal agent runtime built with DSPy modules and a thin Python loop.
- Plan/Act/Finalize expressed as DSPy `Signature`s, with OpenAI-native tool-calling when available.
- Thin runtime (`agent.py`) handles looping, tool routing, and trace persistence.
- CLI and FastAPI server, plus a tiny eval harness.

## Quickstart
- Python 3.10+
- Create a virtualenv and install (using `uv`, or see pip alternative below):
```bash
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env  # set OPENAI_API_KEY or configure Ollama

# Ask a question (append --utc to nudge UTC use when time is relevant)
micro-agent ask --question "What's 2*(3+5)?" --utc

# Run the API server
uvicorn micro_agent.server:app --reload --port 8000

# Run quick evals (repeat small dataset)
python evals/run_evals.py --n 50
```

Pip alternative:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Configuration
- `.env` is loaded automatically (via `python-dotenv`).
- Set one of the following provider configs:
  - OpenAI (default): `OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-4o-mini`)
  - Ollama: `LLM_PROVIDER=ollama`, `OLLAMA_MODEL` (e.g. `llama3.2:1b`), `OLLAMA_HOST` (default `http://localhost:11434`)
- Optional tuning: `TEMPERATURE` (default `0.2`), `MAX_TOKENS` (default `1024`)
- Tool plugins: `TOOLS_MODULES="your_pkg.tools,other_pkg.tools"` to load extra tools (see Tools below)
- Traces location: `TRACES_DIR` (default `traces/`)

Examples:
```bash
# OpenAI
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini

# Ollama
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2:1b
export OLLAMA_HOST=http://localhost:11434
```

## CLI
- `micro-agent ask --question <text> [--utc] [--max-steps N]`
  - `--utc` appends a hint to prefer UTC when time is used.
  - Saves a JSONL trace under `traces/<id>.jsonl` and prints the path.
- `micro-agent replay --path traces/<id>.jsonl [--index -1]`
  - Pretty-prints a saved record from the JSONL file.

Examples:
```bash
micro-agent ask --question "Add 12345 and 67890, then show the current date (UTC)." --utc
micro-agent ask --question "Compute (7**2 + 14)/5 and explain briefly." --max-steps 4
micro-agent replay --path traces/<id>.jsonl --index -1
```

## HTTP API
- Start: `uvicorn micro_agent.server:app --reload --port 8000`
- Endpoint: `POST /ask`
  - Request JSON: `{ "question": "...", "max_steps": 6 }`
  - Response JSON: `{ "answer": str, "trace_id": str, "trace_path": str, "steps": [...] }`

Example:
```bash
curl -s http://localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question":"What\'s 2*(3+5)?","max_steps":6}' | jq .
```

OpenAPI:
- FastAPI publishes `/openapi.json` and interactive docs at `/docs`.
- Schemas reflect `AskRequest` and `AskResponse` models in `micro_agent/server.py`.

## Tools
- Built-ins live in `micro_agent/tools.py`:
  - `calculator`: safe expression evaluator. Supports `+ - * / ** % // ( )` and `!` via rewrite to `fact(n)`.
  - `now`: current timestamp; `{timezone: "utc"|"local"}` (default local).
- Each tool is defined as:
```
Tool(
  "name",
  "description",
  {"type":"object","properties":{...},"required":[...]},
  handler_function,
)
```
- Plugins: set `TOOLS_MODULES` to a comma-separated list of importable modules. Each module should expose either a `TOOLS: dict[str, Tool]` or a `get_tools() -> dict[str, Tool]`.

Runtime validation
- Tool args are validated against the JSON Schema before execution; invalid args add a `⛔️validation_error` step and the agent requests a correction in the next loop. See `micro_agent/tools.py` (run_tool) and `micro_agent/agent.py` (validation error handling).


## Provider Modes
- OpenAI: uses DSPy `PlanWithTools` with `JSONAdapter` to enable native function-calls. The model may return `tool_calls` or a `final` answer; tool calls are executed via our registry.
- Others (e.g., Ollama): uses a robust prompt with few-shot JSON decision demos. Decisions are parsed with strict JSON; on failure we try `json_repair` (if installed) and Python-literal parsing.
- Policy enforcement: if the question implies math, the agent requires a `calculator` step before finalizing; likewise for time/date with the `now` tool. Violations are recorded in the trace as `⛔️policy_violation` steps and planning continues.

Code references (discoverability)
- Replay subcommand: `micro_agent/cli.py` (subparser `replay`, printing JSONL)
- Policy enforcement markers: `micro_agent/agent.py` (look for `⛔️policy_violation` and `⛔️validation_error`)
- Provider fallback and configuration: `micro_agent/config.py` (`configure_lm` tries Ollama → OpenAI → registry fallbacks)
- JSON repair in decision parsing: `micro_agent/runtime.py` (`parse_decision_text` uses `json_repair` if available)

## Tracing
- Each run appends a record to `traces/<id>.jsonl` with fields: `id`, `ts`, `question`, `steps`, `answer`.
- Steps are `{tool, args, observation}` in order of execution.
- Replay: `micro-agent replay --path traces/<id>.jsonl --index -1`.
 - Fetch by id (HTTP): `GET /trace/{id}` (CORS enabled).

## Evals
- Dataset: `evals/tasks.yaml` (small, mixed math/time tasks). Rubric: `evals/rubrics.yaml`.
- Run: `python evals/run_evals.py --n 50`.
- Metrics printed: `success_rate`, `avg_latency_sec`, `avg_lm_calls`, `avg_tool_calls`, `avg_steps`, `n`.

## Optimize (Teleprompting)
- Compile optimized few-shot demos for the OpenAI `PlanWithTools` planner and save to JSON:
```bash
micro-agent optimize --n 12 --tasks evals/tasks.yaml --save opt/plan_demos.json
```
- Apply compiled demos automatically by placing them at the default path or setting:
```bash
export COMPILED_DEMOS_PATH=opt/plan_demos.json
```
- Optional: print a DSPy teleprompting template (for notebooks):
```bash
micro-agent optimize --n 12 --template
```
The agent loads these demos on OpenAI providers and attaches them to the `PlanWithTools` predictor to improve tool selection and output consistency.

## Architecture
- `micro_agent/config.py`: configures DSPy LM. Tries Ollama first if requested, else OpenAI; supports `dspy.Ollama`, `dspy.OpenAI`, and registry fallbacks like `dspy.LM("openai/<model>")`.
- `micro_agent/signatures.py`: DSPy `Signature`s for plan/act/finalize and OpenAI tool-calls.
- `micro_agent/agent.py`: the runtime loop (~100+ LOC). Builds a JSON decision prompt, executes tools, enforces policy, and finalizes.
- `micro_agent/runtime.py`: trace format, persistence, and robust JSON decision parsing utilities.
- `micro_agent/cli.py`: CLI entry (`micro-agent`).
- `micro_agent/server.py`: FastAPI app exposing `POST /ask`.
- `evals/`: tiny harness to sample tasks, capture metrics, and save traces.

## Development
- Make targets: `make init`, `make run`, `make serve`, `make evals`, `make test`.
- Tests: `pytest -q` (note: tests are minimal and do not cover all paths).

## Docker
- Build: `make docker-build`
- Run (OpenAI): `OPENAI_API_KEY=... make docker-run` (maps `:8000`)
- Run (Ollama on host): `make docker-run-ollama` (uses `host.docker.internal:11434`)
- Env (OpenAI): `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-4o-mini`
- Env (Ollama): `LLM_PROVIDER=ollama`, `OLLAMA_HOST=http://host.docker.internal:11434`, `OLLAMA_MODEL=llama3.1:8b`
- Service: `POST http://localhost:8000/ask` and `GET /trace/{id}`

## Compatibility Notes
- DSPy is pinned to `dspy-ai>=2.5.0`. Some adapters (e.g., `JSONAdapter`, `dspy.Ollama`) may vary across versions; the code tries multiple backends and falls back to generic registry forms when needed.
- If `json_repair` is installed, it is used opportunistically to salvage slightly malformed JSON decisions.

## Limitations and Next Steps
- Costs/usage are not recorded; you can plumb LM usage metadata into the eval harness if your wrapper exposes it.
- The finalization step often composes from tool results for reliability; you can swap in a DSPy `Finalize` predictor if preferred.
- Add persistence to a DB instead of JSONL by replacing `dump_trace`.
- Add human-in-the-loop, budgets, retries, or branching per your needs.

## Objective
Prove: an “agent” can be expressed as DSPy modules plus a thin runtime loop.
