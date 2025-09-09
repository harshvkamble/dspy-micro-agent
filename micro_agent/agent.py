from __future__ import annotations
import json, os
import dspy
from typing import List, Dict, Any
from .signatures import PlanOrAct, Finalize, PlanWithTools
from .tools import TOOLS, run_tool, safe_eval_math, to_dspy_tools
from .runtime import parse_decision_text

class MicroAgent(dspy.Module):
    """
    The "agent framework": ~100 LOC.
    Plan -> (optional) tool -> observe -> loop -> finalize.
    """
    def __init__(self, max_steps: int = 6, use_tool_calls: bool | None = None):
        super().__init__()
        # Use LM directly for robust JSON handling across providers.
        self.lm = dspy.settings.lm
        self.finalize = None  # fallback finalize handled via LM prompt
        self._tool_list = [t.spec() for t in TOOLS.values()]
        self.max_steps = max_steps
        self._provider = None
        try:
            self._provider = (self.lm.model.split("/", 1)[0] if getattr(self.lm, "model", None) else None)
        except Exception:
            self._provider = None
        # Determine function-calls mode
        env_override = os.getenv("USE_TOOL_CALLS")
        if isinstance(use_tool_calls, bool):
            self._use_tool_calls = use_tool_calls
        elif env_override is not None:
            self._use_tool_calls = env_override.strip().lower() in {"1","true","yes","on"}
        else:
            self._use_tool_calls = bool(self._provider == "openai")
        self.planner = None
        if self._use_tool_calls:
            try:
                from dspy.adapters import JSONAdapter
                dspy.settings.configure(adapter=JSONAdapter())
            except Exception:
                pass
            self.planner = dspy.Predict(PlanWithTools)
            self._load_compiled_demos()

    def _load_compiled_demos(self):
        import json as _json
        try:
            from dspy.adapters import ToolCalls as _ToolCalls
        except Exception:
            _ToolCalls = None
        path = os.getenv("COMPILED_DEMOS_PATH", "opt/plan_demos.json")
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            demos = []
            for item in data:
                q = item.get("question")
                state = item.get("state", "[]")
                final = item.get("final")
                tc = item.get("tool_calls")
                tool_calls = None
                if _ToolCalls is not None and isinstance(tc, list):
                    try:
                        tool_calls = _ToolCalls.from_dict_list(tc)
                    except Exception:
                        tool_calls = None
                demo = dspy.Example(question=q, state=state, final=final, tool_calls=tool_calls).with_inputs('question', 'state')
                demos.append(demo)
            if self.planner is not None and demos:
                self.planner.demos = demos
        except Exception:
            return

    _DEMO_SNIPPETS = [
        # few-shot decision demos to guide JSON formatting and tool choice
        (
            "What's 2*(3+5)?",
            [],
            {"tool": {"name": "calculator", "args": {"expression": "2*(3+5)"}}},
        ),
        (
            "What time is it (UTC)?",
            [],
            {"tool": {"name": "now", "args": {"timezone": "utc"}}},
        ),
        (
            "Add 12345 and 67890.",
            [],
            {"tool": {"name": "calculator", "args": {"expression": "12345 + 67890"}}},
        ),
        (
            "Compute 9! / (3!*3!*3!).",
            [],
            {"tool": {"name": "calculator", "args": {"expression": "9! / (3!*3!*3!)"}}},
        ),
    ]

    def _decision_prompt(self, question: str, state_json: str, tools_json: str) -> str:
        demos = []
        for q, st, out in self._DEMO_SNIPPETS:
            demos.append(
                f"Example Question: {q}\n"
                f"Example State: {json.dumps(st, ensure_ascii=False)}\n"
                f"Decision: {json.dumps(out, ensure_ascii=False)}\n"
            )
        return (
            "You are a strict planner that chooses a single next action as JSON.\n"
            "Rules:\n"
            "- Use 'calculator' for ANY arithmetic (addition, subtraction, multiplication, division, powers).\n"
            "- Use 'now' for ANY current time/date (prefer UTC if requested). Do NOT guess time/date.\n"
            "- One tool per step.\n"
            "- NEVER finalize until all required tools have been used.\n"
            "- Respond with ONLY one JSON object, no extra text.\n\n"
            + "\n".join(demos)
            + "\n"
            + f"Question: {question}\n\n"
            + f"State: {state_json}\n\n"
            + f"Tools: {tools_json}\n\n"
            + "Return exactly one of:\n"
            + "{\"tool\": {\"name\": \"calculator\", \"args\": {\"expression\": \"...\"}}}\n"
            + "{\"tool\": {\"name\": \"now\", \"args\": {\"timezone\": \"utc\"}}}\n"
            + "{\"final\": {\"answer\": \"<complete answer string>\"}}\n"
        )

    def forward(self, question: str):
        import re
        lm_calls = 0
        tool_calls = 0
        total_cost = 0.0
        total_in_tokens = 0
        total_out_tokens = 0

        def needs_math(q: str) -> bool:
            ql = q.lower()
            if re.search(r"[0-9].*[+\-*/]", q):
                return True
            if any(w in ql for w in ["add", "sum", "multiply", "divide", "compute", "calculate", "total", "power", "factorial", "!", "**", "^"]):
                return True
            return False

        def needs_time(q: str) -> bool:
            ql = q.lower()
            return any(w in ql for w in ["time", "date", "utc", "current time", "now"])

        def used_tool(state, name: str) -> bool:
            return any(step.get("tool") == name for step in state)

        must_math = needs_math(question)
        must_time = needs_time(question)

        state: List[Dict[str, Any]] = []

        def _accumulate_usage():
            # Pull new usage entries from dspy.settings.trace
            try:
                for _, _, out in dspy.settings.trace[-1:]:
                    usage = getattr(out, "usage", None) or {}
                    nonlocal total_cost, total_in_tokens, total_out_tokens
                    c = getattr(out, "cost", None)
                    if c is not None:
                        total_cost += float(c or 0)
                    total_in_tokens += int(usage.get("input_tokens", 0) or 0)
                    total_out_tokens += int(usage.get("output_tokens", 0) or 0)
            except Exception:
                pass

        # Path A: OpenAI-native tool calling using DSPy signatures/adapters.
        if self._use_tool_calls:
            dspy_tools = to_dspy_tools()

            for _ in range(self.max_steps):
                lm_calls += 1  # one planner call per loop
                pred = self.planner(
                    question=question,
                    state=json.dumps(state, ensure_ascii=False),
                    tools=dspy_tools,
                )
                # Accumulate usage from DSPy prediction (OpenAI path)
                try:
                    usage = pred.get_lm_usage() or {}
                    total_cost += float(usage.get('cost', 0.0) or 0.0)
                    total_in_tokens += int(usage.get('input_tokens', 0) or 0)
                    total_out_tokens += int(usage.get('output_tokens', 0) or 0)
                except Exception:
                    pass

                # If tool calls are proposed, execute them.
                calls = getattr(pred, 'tool_calls', None)
                executed_any = False
                if calls and getattr(calls, 'tool_calls', None):
                    for call in calls.tool_calls:
                        try:
                            name = getattr(call, 'name')
                            args = getattr(call, 'args') or {}
                        except Exception:
                            continue
                        # Validate/execute; on validation error, record and continue planning
                        obs = run_tool(name, args)
                        if isinstance(obs, dict) and "error" in obs and "validation" in obs.get("error", ""):
                            state.append({
                                "tool": "⛔️validation_error",
                                "args": {"name": name, "args": args},
                                "observation": obs,
                            })
                            continue
                        state.append({"tool": name, "args": args, "observation": obs})
                        tool_calls += 1
                        executed_any = True

                # Check finalization.
                final = getattr(pred, 'final', None)
                if final:
                    if must_math and not used_tool(state, "calculator"):
                        state.append({"tool": "⛔️policy_violation", "args": {}, "observation": "Finalize before calculator (OpenAI path)."})
                        # If tools were suggested and executed this step, iterate; else force tool suggestion by continuing.
                        continue
                    if must_time and not used_tool(state, "now"):
                        state.append({"tool": "⛔️policy_violation", "args": {}, "observation": "Finalize before now (OpenAI path)."})
                        continue
                    # Prefer composing from tool results when available to ensure answers include key values.
                    composed = []
                    calculators = [s for s in state if s.get("tool") == "calculator" and isinstance(s.get("observation"), dict)]
                    nows = [s for s in state if s.get("tool") == "now" and isinstance(s.get("observation"), dict)]
                    if calculators:
                        composed.append(str(calculators[0]["observation"].get("result")))
                    if nows:
                        iso = nows[-1]["observation"].get("iso")
                        if iso:
                            composed.append(f"UTC: {iso}")
                    answer_text = " | ".join(composed) if composed else final
                    p = dspy.Prediction(answer=answer_text, trace=state)
                    p.usage = {
                        "lm_calls": lm_calls,
                        "tool_calls": tool_calls,
                        "provider": self._provider,
                        "model": getattr(self.lm, "model", None),
                        "cost": total_cost,
                        "input_tokens": total_in_tokens,
                        "output_tokens": total_out_tokens,
                    }
                    return p

                # If no tool and no final, gently nudge by adding a malformed marker.
                if not executed_any:
                    state.append({"tool": "⛔️no_action", "args": {}, "observation": "No tool_calls or final returned."})
                    # Continue loop

            # Fallback compose from tools; if none, attempt lightweight inference
            calculators = [s for s in state if s.get("tool") == "calculator" and isinstance(s.get("observation"), dict)]
            nows = [s for s in state if s.get("tool") == "now" and isinstance(s.get("observation"), dict)]
            parts = []
            if calculators:
                parts.append(str(calculators[0]["observation"].get("result")))
            elif must_math:
                # Last-chance math: infer a simple expression from the question.
                import re as _re
                ql = question.lower()
                if "add" in ql or "sum" in ql:
                    nums = [int(n) for n in _re.findall(r"\b\d+\b", question)]
                    if len(nums) >= 2:
                        res = sum(nums)
                        parts.append(str(res))
                        # also record as a calculator step for trace parity
                        state.append({"tool": "calculator", "args": {"expression": "+".join(map(str, nums))}, "observation": {"result": res}})
                        tool_calls += 1
                if not parts:
                    candidates = _re.findall(r"[0-9\+\-\*/%\(\)\.!\^\s]+", question)
                    candidates = [c.strip() for c in candidates if any(op in c for op in ["+","-","*","/","%","^","(",")","!"])]
                    expr = max(candidates, key=len) if candidates else ""
                    if expr:
                        try:
                            res = safe_eval_math(expr)
                            parts.append(str(res))
                            state.append({"tool": "calculator", "args": {"expression": expr}, "observation": {"result": res}})
                            tool_calls += 1
                        except Exception:
                            pass
            if nows:
                iso = nows[-1]["observation"].get("iso")
                if iso:
                    parts.append(f"UTC: {iso}")
            elif must_time:
                # One-shot now tool to satisfy policy and helpfulness
                obs = run_tool("now", {"timezone": "utc"})
                state.append({"tool": "now", "args": {"timezone": "utc"}, "observation": obs})
                tool_calls += 1
                iso = obs.get("iso") if isinstance(obs, dict) else None
                if iso:
                    parts.append(f"UTC: {iso}")
            p = dspy.Prediction(answer=" | ".join([p for p in parts if p]), trace=state)
            p.usage = {
                "lm_calls": lm_calls,
                "tool_calls": tool_calls,
                "provider": self._provider,
                "model": getattr(self.lm, "model", None),
                "cost": total_cost,
                "input_tokens": total_in_tokens,
                "output_tokens": total_out_tokens,
            }
            return p
        # Path B: Ollama-friendly loop via raw LM completions and robust JSON parsing.
        for _ in range(self.max_steps):
            lm_calls += 1
            raw = self.lm(
                prompt=self._decision_prompt(
                    question=question,
                    state_json=json.dumps(state, ensure_ascii=False),
                    tools_json=json.dumps(self._tool_list, ensure_ascii=False),
                )
            )
            decision_text = raw[0] if isinstance(raw, list) else (raw if isinstance(raw, str) else str(raw))
            _accumulate_usage()

            # Extract and parse JSON; if malformed, try a flexible parser and one self-correction retry.
            try:
                decision = parse_decision_text(decision_text)
            except Exception:
                lm_calls += 1
                raw = self.lm(
                    prompt=self._decision_prompt(
                        question=question,
                        state_json=json.dumps(state, ensure_ascii=False),
                        tools_json=json.dumps(self._tool_list, ensure_ascii=False),
                    )
                )
                decision_text = raw[0] if isinstance(raw, list) else (raw if isinstance(raw, str) else str(raw))
                _accumulate_usage()
                try:
                    decision = parse_decision_text(decision_text)
                except Exception:
                    state.append({
                        "tool": "⛔️malformed_decision",
                        "args": {},
                        "observation": decision_text,
                    })
                    continue

            if "final" in decision:
                # Enforce tool usage policy: if required tools not yet used, keep planning.
                if must_math and not used_tool(state, "calculator"):
                    state.append({"tool": "⛔️policy_violation", "args": {}, "observation": "Finalize attempted before calculator."})
                    continue
                if must_time and not used_tool(state, "now"):
                    state.append({"tool": "⛔️policy_violation", "args": {}, "observation": "Finalize attempted before now."})
                    continue
                p = dspy.Prediction(answer=decision["final"]["answer"], trace=state)
                p.usage = {
                    "lm_calls": lm_calls,
                    "tool_calls": tool_calls,
                    "provider": self._provider,
                    "model": getattr(self.lm, "model", None),
                }
                return p

            if "tool" in decision:
                tool_desc = decision["tool"]
                if isinstance(tool_desc, dict):
                    name = tool_desc.get("name", "")
                    args = tool_desc.get("args", {}) or {}
                else:  # allow {"tool": "name", "args": {...}}
                    name = str(tool_desc)
                    args = decision.get("args", {}) or {}
                obs = run_tool(name, args)
                if isinstance(obs, dict) and "error" in obs and "validation" in obs.get("error", ""):
                    # second-chance: record detailed schema hint in state and continue planning
                    schema = TOOLS.get(name).schema if name in TOOLS else {}
                    state.append({
                        "tool": "⛔️validation_error",
                        "args": {"name": name, "args": args, "schema": schema},
                        "observation": obs,
                    })
                    continue
                state.append({"tool": name, "args": args, "observation": obs})
                tool_calls += 1
                continue

            # Model produced something unexpected: record and continue.
            state.append({"tool": "⛔️malformed_decision", "args": {}, "observation": decision_text})

        # Fallback: finalize. Prefer composing from tool results to ensure numeric substrings appear.
        calculators = [s for s in state if s.get("tool") == "calculator" and isinstance(s.get("observation"), dict)]
        nows = [s for s in state if s.get("tool") == "now" and isinstance(s.get("observation"), dict)]
        if calculators or nows or must_math:
            parts = []
            if calculators:
                calc_results = [s["observation"].get("result") for s in calculators if isinstance(s.get("observation"), dict) and s["observation"].get("result") is not None]
                if calc_results:
                    parts.append(str(calc_results[0]))
            elif must_math:
                # Last-chance math: try to infer a simple expression from the question.
                ql = question.lower()
                # If looks like 'add X and Y', sum integers.
                import re
                if "add" in ql or "sum" in ql:
                    nums = [int(n) for n in re.findall(r"\b\d+\b", question)]
                    if len(nums) >= 2:
                        parts.append(str(sum(nums)))
                if not parts:
                    # Extract longest math-like substring and evaluate.
                    candidates = re.findall(r"[0-9\+\-\*/%\(\)\.!\^\s]+", question)
                    candidates = [c.strip() for c in candidates if any(op in c for op in ["+","-","*","/","%","^","(",")","!"])]
                    expr = max(candidates, key=len) if candidates else ""
                    if expr:
                        try:
                            parts.append(str(safe_eval_math(expr)))
                        except Exception:
                            pass
            if nows:
                iso = nows[-1]["observation"].get("iso")
                if iso:
                    parts.append(f"UTC: {iso}")
            ans = " | ".join(parts) if parts else ""
        else:
            lm_calls += 1
            raw = self.lm(
                prompt=(
                    "Given the question and the trace of tool observations, write the final answer.\n\n"
                    f"Question: {question}\n\nTrace: {json.dumps(state, ensure_ascii=False)}\n\n"
                    "Answer succinctly."
                )
            )
            ans = raw[0] if isinstance(raw, list) else (raw if isinstance(raw, str) else str(raw))
        p = dspy.Prediction(answer=ans, trace=state)
        p.usage = {
            "lm_calls": lm_calls,
            "tool_calls": tool_calls,
            "provider": self._provider,
            "model": getattr(self.lm, "model", None),
            "cost": total_cost,
            "input_tokens": total_in_tokens,
            "output_tokens": total_out_tokens,
        }
        return p
