from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from .config import configure_lm
from .agent import MicroAgent
from .runtime import dump_trace, new_trace_id

app = FastAPI(title="DSPy Micro Agent")

class AskRequest(BaseModel):
    question: str
    max_steps: int = 6

class AskResponse(BaseModel):
    answer: str
    trace_id: str
    trace_path: str
    steps: list

configure_lm()
_agent = MicroAgent()

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    pred = _agent(req.question)
    trace_id = new_trace_id()
    path = dump_trace(trace_id, req.question, pred.trace, pred.answer)
    return AskResponse(answer=pred.answer, trace_id=trace_id, trace_path=path, steps=pred.trace)

