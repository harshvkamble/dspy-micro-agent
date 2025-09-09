from __future__ import annotations
"""
Optimize (teleprompting) stub for DSPy.

This provides a baseline-eval + guidance to run DSPy teleprompting with OpenAI.

Why a stub? Teleprompting works best when you curate training examples and a
metric that matches your domain. We include a ready template to adapt.
"""

import argparse
from statistics import mean
import time
import json
import os
from typing import List, Dict, Any

import dspy

from .config import configure_lm
from .agent import MicroAgent


def _run_quick_eval(n: int = 12, tasks_path: str = "evals/tasks.yaml") -> Dict[str, Any]:
    import yaml
    from .runtime import new_trace_id, dump_trace

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = yaml.safe_load(f)

    # Repeat/trim to n
    dataset = (tasks * ((n + len(tasks) - 1) // len(tasks)))[:n]

    agent = MicroAgent(max_steps=6)
    scores, latencies = [], []
    for i, item in enumerate(dataset, 1):
        q = item["question"]
        t0 = time.time()
        pred = agent(q)
        dt = time.time() - t0
        latencies.append(dt)
        trace_id = new_trace_id()
        dump_trace(trace_id, q, pred.trace, pred.answer)
        expect = item.get("expect_contains")
        s = int(expect in pred.answer) if expect else 1
        scores.append(s)

    return {
        "success_rate": mean(scores) if scores else 0.0,
        "avg_latency_sec": mean(latencies) if latencies else 0.0,
        "n": len(dataset),
    }


TEMPLATE = """
Example: Teleprompt the OpenAI tool-calling planner

import dspy
from dspy.teleprompt import BootstrapFewShot
from micro_agent.signatures import PlanWithTools
from micro_agent.tools import to_dspy_tools

class Planner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.decide = dspy.Predict(PlanWithTools)

    def forward(self, question: str, state: str = "[]"):
        tools = to_dspy_tools()
        return self.decide(question=question, state=state, tools=tools)

def metric(example, pred) -> float:
    # Reward tool_calls for math/time; reward final containing expectation
    import json
    q = example["question"]
    expect = example.get("expect_contains")
    score = 0.0
    # Encourage tool usage for questions with numbers or time
    if any(ch.isdigit() for ch in q) and getattr(pred, 'tool_calls', None):
        score += 0.5
    if "time" in q.lower() or "utc" in q.lower():
        if getattr(pred, 'tool_calls', None):
            score += 0.5
    # If a final was produced, check expectation
    final = getattr(pred, 'final', '') or ''
    if expect and expect in str(final):
        score += 1.0
    return score

tp = BootstrapFewShot(metric=metric, max_bootstrapped_demos=8)
optimized = tp.compile(Planner())

# Use `optimized` inside your agent in place of the ad-hoc planner
"""


def optimize_cli(args: argparse.Namespace):
    configure_lm()
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    print(f"Provider: {provider}")
    print("Running a quick baseline eval before optimization...")
    baseline = _run_quick_eval(n=args.n, tasks_path=args.tasks)
    print("Baseline:")
    print(json.dumps(baseline, indent=2))

    if provider != "openai":
        print("\nNote: Teleprompting template is optimized for OpenAI providers.")

    print("\n=== Teleprompting Template (copy/paste into a notebook) ===\n")
    print(TEMPLATE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--tasks", default="evals/tasks.yaml")
    args = p.parse_args()
    optimize_cli(args)


if __name__ == "__main__":
    main()

