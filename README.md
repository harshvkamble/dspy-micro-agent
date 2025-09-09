# DSPy Micro Agent

Minimal agent runtime using **DSPy** modules.
- Plan/Act/Finalize implemented as DSPy `Signature`s.
- Thin Python loop for tool routing and trace persistence.
- CLI + FastAPI server.
- Eval harness.

## Quickstart
```bash
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env  # set OPENAI_API_KEY
micro-agent ask --question "What's 2*(3+5)? Use UTC for time."
uvicorn micro_agent.server:app --reload --port 8000
python evals/run_evals.py --n 50
```

## Replace or extend tools

Edit `micro_agent/tools.py`. Each tool:

```
Tool(
  "name",
  "description",
  {"type":"object","properties":{...},"required":[...]},
  handler_function
)
```

---

## How it proves the point

- Planning + acting: `PlanOrAct` and `Finalize` are pure DSPy modules.
- “Agent framework”: The runtime is ~100 LOC in `agent.py` (loop, tool execution, trace list).
- Observability: JSONL traces on disk; easy to ship into your logging stack.
- Evals: dataset + rubric shows measurable success/latency; swap in your real tasks.

---

## Extension hooks (optional)

- Durability: replace `dump_trace` with a DB (SQLite/Postgres). Add a `checkpoint(state)` every step.
- Human‑in‑the‑loop: insert a pause after selected steps; await approval before continuing.
- Parallel branches: spawn multiple tool calls by forking the loop state; merge with a simple reducer.
- Budgets: add counters for token/cost ceilings; if exceeded, force finalize.
- Retry policy: wrap `run_tool` with backoff + jitter; add circuit‑breaker flags in the state.

---

## Unknowns to resolve (labelled)

1) DSPy version you’ll pin to (APIs occasionally shift).  
2) Model provider you prefer (OpenAI default here; Anthropic config is a 5‑line tweak in `config.py` if your DSPy build supports `dspy.Anthropic`).

---

### Suggested first run (copy/paste)

```bash
# fresh dir
mkdir dspy-micro-agent && cd dspy-micro-agent
# paste files according to the structure above (or hand this spec to your code assistant)
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env && $EDITOR .env   # set OPENAI_API_KEY

micro-agent ask --question "Compute (7**2 + 14)/5 and explain briefly; prefer UTC time if used."
python evals/run_evals.py --n 24
uvicorn micro_agent.server:app --reload --port 8000
```

## Improvements inspired by DSPy tutorials

- Provider-aware adapters: OpenAI can use DSPy `Predict(Signature)` with JSON adapters; Ollama path uses direct LM prompts + robust JSON repair when models drift.
- Few-shot guidance: The planner prompt includes compact JSON decision demos to stabilize tool selection and formatting.
- Teleprompt-ready: You can plug in `dspy.teleprompt` (e.g., `BootstrapFewShot`) to optimize signatures when using OpenAI providers.
- Extensible tools: Load extra tool modules by setting `TOOLS_MODULES="your_pkg.tools,other_pkg.tools"` (each module exposes a `TOOLS` dict or `get_tools()`).
- Replay: `micro-agent replay --path traces/<id>.jsonl` prints the saved run for inspection or debugging.

### Using Ollama
```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.1:8b   # pick an installed model
micro-agent ask --question "Add 12345 and 67890, then UTC date?" --utc
```

### Using OpenAI
```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
micro-agent ask --question "Compute (7**2+14)/5 and explain briefly"
```


---

## Objective

Prove: “An agent is just DSPy modules + a thin runtime loop.”
