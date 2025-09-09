from __future__ import annotations
import ast, operator, datetime, math, re
from typing import List
from typing import Any, Dict, Callable
from jsonschema import validate as _js_validate, ValidationError as _JSValidationError

class Tool:
    def __init__(self, name: str, description: str, schema: Dict[str, Any], func: Callable[[Dict[str, Any]], Any]):
        self.name = name
        self.description = description
        self.schema = schema
        self.func = func

    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "schema": self.schema}

# --- Safe calculator ---
MAX_ALLOWED_OPS_NODES = 200
MAX_EXPONENT = 12
MAX_ABS_NUMBER = 10**12
MAX_FACTORIAL_N = 12

ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos
}
ALLOWED_CALLS = {"fact": lambda x: math.factorial(int(x))}
def _eval_expr(node):
    # Python 3.10+: numeric literals appear as ast.Constant
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPS:
        lv, rv = _eval_expr(node.left), _eval_expr(node.right)
        if isinstance(lv, (int, float)) and abs(lv) > MAX_ABS_NUMBER: raise ValueError("number too large")
        if isinstance(rv, (int, float)) and abs(rv) > MAX_ABS_NUMBER: raise ValueError("number too large")
        if isinstance(node.op, ast.Pow):
            if isinstance(rv, (int, float)) and abs(rv) > MAX_EXPONENT:
                raise ValueError("exponent too large")
        return ALLOWED_OPS[type(node.op)](lv, rv)
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPS:
        v = _eval_expr(node.operand)
        if isinstance(v, (int, float)) and abs(v) > MAX_ABS_NUMBER: raise ValueError("number too large")
        return ALLOWED_OPS[type(node.op)](v)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ALLOWED_CALLS:
        if len(node.args) != 1:
            raise ValueError("Invalid arguments")
        arg = _eval_expr(node.args[0])
        if isinstance(arg, (int, float)) and arg > MAX_FACTORIAL_N:
            raise ValueError("factorial too large")
        return ALLOWED_CALLS[node.func.id](arg)
    if isinstance(node, ast.Expression): return _eval_expr(node.body)
    raise ValueError("Disallowed expression")

def preprocess_math(expr: str) -> str:
    # Replace simple factorial forms like 9! or 12! with fact(9) / fact(12)
    expr = re.sub(r"(\d+)\!", r"fact(\1)", expr)
    # Replace caret ^ with exponentiation
    expr = expr.replace("^", "**")
    return expr

def safe_eval_math(expr: str) -> float:
    expr = preprocess_math(expr)
    tree = ast.parse(expr, mode="eval")
    # cap complexity
    if sum(1 for _ in ast.walk(tree)) > MAX_ALLOWED_OPS_NODES:
        raise ValueError("expression too complex")
    return _eval_expr(tree)

def tool_calculator(args: Dict[str, Any]):
    expr = str(args.get("expression", "")).strip()
    return {"result": safe_eval_math(expr)}

def tool_now(args: Dict[str, Any]):
    tz = str(args.get("timezone", "local")).lower()
    now = datetime.datetime.now(datetime.timezone.utc) if tz == "utc" else datetime.datetime.now()
    return {"iso": now.isoformat(timespec="seconds")}

def _load_plugins():
    import importlib, os
    mods = os.getenv("TOOLS_MODULES", "").strip()
    if not mods:
        return {}
    tools = {}
    for m in [x.strip() for x in mods.split(",") if x.strip()]:
        try:
            mod = importlib.import_module(m)
        except Exception as e:
            continue
        if hasattr(mod, "TOOLS") and isinstance(getattr(mod, "TOOLS"), dict):
            tools.update(getattr(mod, "TOOLS"))
        elif hasattr(mod, "get_tools"):
            try:
                tools.update(mod.get_tools())
            except Exception:
                pass
    return tools

TOOLS = {
    "calculator": Tool(
        "calculator",
        "Evaluate arithmetic expressions. Schema: {expression: string}. Supports +,-,*,/,**,%, //, parentheses.",
        {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        tool_calculator
    ),
    "now": Tool(
        "now",
        "Return the current timestamp. Optional: {timezone: 'utc'|'local'}",
        {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": []},
        tool_now
    ),
}
TOOLS.update(_load_plugins())

def to_dspy_tools() -> List["DSpyTool"]:
    """Convert registry tools to DSPy Tool objects for native function calling.

    Returns a list of dspy.adapters.Tool objects whose names/args mirror our registry.
    """
    try:
        from dspy.adapters import Tool as DSpyTool  # lazy import; optional for Ollama path
    except Exception:
        return []

    dtools: List[DSpyTool] = []
    for name, t in TOOLS.items():
        try:
            props = (t.schema or {}).get("properties", {})
            desc = t.description or name
            # Provide a lightweight callable; not used by adapter execution path.
            def _dummy(**kwargs):
                return None
            dtools.append(DSpyTool(func=_dummy, name=name, desc=desc, args=props))
        except Exception:
            continue
    return dtools

def run_tool(name: str, args: Dict[str, Any]):
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'"}
    try:
        # Validate against JSON Schema when provided
        schema = TOOLS[name].schema or {}
        if schema:
            try:
                _js_validate(instance=args or {}, schema=schema)
            except _JSValidationError as e:
                raise ValueError(f"args validation failed: {e.message}")
        return TOOLS[name].func(args or {})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
