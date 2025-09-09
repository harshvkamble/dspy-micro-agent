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

    scores, latencies, costs = [], [], []  # costs optional; if LM exposes usage, record it.

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
        # If your LM wrapper exposes usage/cost, add it here. DSPy wrappers often attach metadata.

        print(f"[{i}/{len(dataset)}] s={s} t={dt:.2f}s  q={q!r}")

    print("\n=== METRICS ===")
    print(json.dumps({
        "success_rate": mean(scores),
        "avg_latency_sec": mean(latencies),
        "n": len(dataset),
    }, indent=2))

if __name__ == "__main__":
    main()

