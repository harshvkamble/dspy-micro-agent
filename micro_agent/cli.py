from __future__ import annotations
import argparse, json, os
from rich.console import Console
from rich.panel import Panel
from .config import configure_lm
from .agent import MicroAgent
from .runtime import dump_trace, new_trace_id

console = Console()

def main():
    parser = argparse.ArgumentParser(prog="micro-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="Ask a question via the micro agent")
    a.add_argument("--question", required=True)
    a.add_argument("--utc", action="store_true", help="Ask time in UTC when using the 'now' tool")
    a.add_argument("--max-steps", type=int, default=6)

    args = parser.parse_args()

    configure_lm()
    agent = MicroAgent(max_steps=args.max_steps)

    q = args.question
    if args.utc and "UTC" not in q:
        q += " (If using time, prefer UTC.)"

    pred = agent(q)
    trace_id = new_trace_id()
    path = dump_trace(trace_id, q, pred.trace, pred.answer)

    console.print(Panel.fit(pred.answer, title="ANSWER"))
    console.print()
    console.print(Panel.fit(json.dumps(pred.trace, indent=2, ensure_ascii=False), title=f"TRACE (saved: {path})"))

