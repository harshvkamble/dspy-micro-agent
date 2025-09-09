from __future__ import annotations
import argparse, json, os
from rich.console import Console
from rich.panel import Panel
from .config import configure_lm
from .agent import MicroAgent
from .runtime import dump_trace, new_trace_id
from .costs import estimate_prediction_cost

console = Console()

def main():
    parser = argparse.ArgumentParser(prog="micro-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="Ask a question via the micro agent")
    a.add_argument("--question", required=True)
    a.add_argument("--utc", action="store_true", help="Ask time in UTC when using the 'now' tool")
    a.add_argument("--max-steps", type=int, default=6)
    g = a.add_mutually_exclusive_group()
    g.add_argument("--func-calls", action="store_true", help="Force OpenAI function-calls mode when available")
    g.add_argument("--no-func-calls", action="store_true", help="Disable function-calls; use robust JSON planning")

    r = sub.add_parser("replay", help="Replay a saved trace JSONL record (by path)")
    r.add_argument("--path", required=True, help="Path to a trace .jsonl file")
    r.add_argument("--index", type=int, default=-1, help="Record index in the file (default last)")

    o = sub.add_parser("optimize", help="Compile optimized planner demos and/or print a DSPy teleprompting template")
    o.add_argument("--n", type=int, default=12)
    o.add_argument("--tasks", default="evals/tasks.yaml")
    o.add_argument("--save", default="opt/plan_demos.json", help="Path to save compiled demos (JSON)")
    o.add_argument("--template", action="store_true", help="Also print a teleprompting template")

    args = parser.parse_args()

    configure_lm()
    if args.cmd == "replay":
        from rich.syntax import Syntax
        with open(args.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            console.print("No records in file.")
            return
        idx = args.index if args.index >= 0 else len(lines)-1
        rec = json.loads(lines[idx])
        console.print(Panel.fit(Syntax(json.dumps(rec, indent=2, ensure_ascii=False), "json"), title=f"REPLAY {idx} : {args.path}"))
        return

    if args.cmd == "optimize":
        from . import optimize as opt
        opt.optimize_cli(args)
        return

    use_tool_calls = None
    if args.func_calls:
        use_tool_calls = True
    elif args.no_func_calls:
        use_tool_calls = False
    agent = MicroAgent(max_steps=args.max_steps, use_tool_calls=use_tool_calls)

    q = args.question
    if args.utc and "UTC" not in q:
        q += " (If using time, prefer UTC.)"

    pred = agent(q)
    trace_id = new_trace_id()
    usage = getattr(pred, "usage", {}) or {}
    est = estimate_prediction_cost(q, pred.trace, pred.answer, usage)
    path = dump_trace(trace_id, q, pred.trace, pred.answer, usage=usage, cost_usd=est.get("cost_usd"))

    console.print(Panel.fit(pred.answer, title="ANSWER"))
    console.print()
    console.print(Panel.fit(json.dumps({"usage": usage, "estimates": est}, indent=2, ensure_ascii=False), title="USAGE / ESTIMATES"))
    console.print()
    console.print(Panel.fit(json.dumps(pred.trace, indent=2, ensure_ascii=False), title=f"TRACE (saved: {path})"))
