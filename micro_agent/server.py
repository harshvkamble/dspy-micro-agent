from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, json
from pydantic import BaseModel
from .config import configure_lm
from .agent import MicroAgent
from .runtime import dump_trace, new_trace_id

app = FastAPI(title="DSPy Micro Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    max_steps: int = 6
    use_tool_calls: bool | None = None

class AskResponse(BaseModel):
    answer: str
    trace_id: str
    trace_path: str
    steps: list

configure_lm()
_agent = MicroAgent()

@app.get("/health")
def health():
    lm = _agent.lm
    return {
        "status": "ok",
        "provider": getattr(_agent, "_provider", None),
        "model": getattr(lm, "model", None),
        "max_steps": getattr(_agent, "max_steps", None),
    }

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    agent = _agent if req.use_tool_calls is None and req.max_steps == _agent.max_steps else MicroAgent(max_steps=req.max_steps, use_tool_calls=req.use_tool_calls)
    pred = agent(req.question)
    trace_id = new_trace_id()
    path = dump_trace(trace_id, req.question, pred.trace, pred.answer)
    return AskResponse(answer=pred.answer, trace_id=trace_id, trace_path=path, steps=pred.trace)

@app.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    traces_dir = os.getenv("TRACES_DIR", "traces")
    path = os.path.join(traces_dir, f"{trace_id}.jsonl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Trace not found")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return {"trace_id": trace_id, "path": path, "records": out}
