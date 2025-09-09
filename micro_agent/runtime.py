from __future__ import annotations
import json, os, re, time, uuid, datetime
from typing import Any, Dict, List, Optional, TypedDict, NotRequired
import ast
try:
    import json_repair
except Exception:
    json_repair = None

TRACES_DIR = os.getenv("TRACES_DIR", "traces")
os.makedirs(TRACES_DIR, exist_ok=True)

class Step(TypedDict):
    tool: str
    args: Dict[str, Any]
    observation: Any

class TraceRecord(TypedDict):
    id: str
    ts: str
    question: str
    steps: List[Step]
    answer: str
    usage: NotRequired[Dict[str, Any]]
    cost_usd: NotRequired[float]

def new_trace_id() -> str:
    return uuid.uuid4().hex

def dump_trace(trace_id: str, question: str, steps: List[Step], answer: str, *, usage: Optional[Dict[str, Any]] = None, cost_usd: Optional[float] = None) -> str:
    rec: TraceRecord = {
        "id": trace_id,
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "steps": steps,
        "answer": answer,
    }
    if usage is not None:
        rec["usage"] = usage
    if cost_usd is not None:
        rec["cost_usd"] = float(cost_usd)
    path = os.path.join(TRACES_DIR, f"{trace_id}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path

_JSON_RE = re.compile(r"\{.*\}", re.S)

def extract_json_block(text: str) -> str:
    """
    Extract the first {...} block to survive models adding prose or code fences.
    """
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object found in: {text[:200]!r}")
    return m.group(0)

def parse_decision_text(text: str) -> Dict[str, Any]:
    """Parse a model decision string into a dict.

    Strategy:
    1) Extract first {...} block.
    2) Try strict JSON parse.
    3) Fallback to Python literal parse (single quotes, etc.).
    """
    block = extract_json_block(text)
    # 1) strict json
    try:
        return json.loads(block)
    except Exception:
        pass
    # 2) json-repair (if available)
    if json_repair is not None:
        try:
            repaired = json_repair.repair(block)
            return json.loads(repaired)
        except Exception:
            pass
    # 3) python literal (handles single quotes)
    try:
        obj = ast.literal_eval(block)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    raise ValueError("Could not parse decision as JSON or Python literal")

def now_iso(utc: bool = False) -> str:
    if utc:
        return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return datetime.datetime.now().isoformat(timespec="seconds")
