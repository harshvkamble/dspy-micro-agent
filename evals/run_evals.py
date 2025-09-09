from __future__ import annotations
import time, json, argparse, yaml
from statistics import mean
from micro_agent.config import configure_lm
from micro_agent.agent import MicroAgent
from micro_agent.runtime import new_trace_id, dump_trace

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def score_answer(answer: str, item: dict) -> int:
    # Binary scoring for now: 1 if substring present, else 0
    expect = item.get("expect_contains")
    if expect is None:
        return 1  # no constraint = pass
    return int(expect in answer)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="evals/tasks.yaml")
    p.add_argument("--rubric", default="evals/rubrics.yaml")
    p.add_argument("--n", type=int, default=50, help="Repeat dataset N/len(tasks) times total.")
    p.add_argument("--max-steps", type=int, default=6)
    args = p.parse_args()

    tasks = load_yaml(args.tasks)
    rub = load_yaml(args.rubric)

    configure_lm()
    agent = MicroAgent(max_steps=args.max_steps)

    # Expand to ~N items
    dataset = (tasks * ((args.n + len(tasks) - 1) // len(tasks)))[:args.n]

    scores, latencies = [], []
    lm_calls_list, tool_calls_list, steps_list = [], [], []

    for i, item in enumerate(dataset, 1):
        q = item["question"]
        t0 = time.time()
        pred = agent(q)
        dt = time.time() - t0

        trace_id = new_trace_id()
        dump_trace(trace_id, q, pred.trace, pred.answer)

        s = score_answer(pred.answer, item)
        scores.append(s)
        latencies.append(dt)
        # Basic usage tracking (provided by MicroAgent)
        usage = getattr(pred, "usage", {}) or {}
        lm_calls_list.append(int(usage.get("lm_calls", 0) or 0))
        tool_calls_list.append(int(usage.get("tool_calls", 0) or 0))
        steps_list.append(len(pred.trace or []))

        print(f"[{i}/{len(dataset)}] s={s} t={dt:.2f}s  q={q!r}")

    print("\n=== METRICS ===")
    print(json.dumps({
        "success_rate": mean(scores) if scores else 0.0,
        "avg_latency_sec": mean(latencies) if latencies else 0.0,
        "avg_lm_calls": mean(lm_calls_list) if lm_calls_list else 0.0,
        "avg_tool_calls": mean(tool_calls_list) if tool_calls_list else 0.0,
        "avg_steps": mean(steps_list) if steps_list else 0.0,
        "n": len(dataset),
    }, indent=2))

if __name__ == "__main__":
    main()
