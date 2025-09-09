from __future__ import annotations
import json
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
    def __init__(self, max_steps: int = 6):
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
        self._use_tool_calls = bool(self._provider == "openai")

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

        # Path A: OpenAI-native tool calling using DSPy signatures/adapters.
        if self._use_tool_calls:
            # Temporarily set JSONAdapter to enable native function calling.
            orig_adapter = dspy.settings.get('adapter', None)
            try:
                from dspy.adapters import JSONAdapter
                dspy.settings.configure(adapter=JSONAdapter())
            except Exception:
                pass

            dspy_tools = to_dspy_tools()
            planner = dspy.Predict(PlanWithTools)

            for _ in range(self.max_steps):
                lm_calls += 1  # one planner call per loop
                pred = planner(
                    question=question,
                    state=json.dumps(state, ensure_ascii=False),
                    tools=dspy_tools,
                )

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
                        obs = run_tool(name, args)
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
                    p = dspy.Prediction(answer=final, trace=state)
                    p.usage = {
                        "lm_calls": lm_calls,
                        "tool_calls": tool_calls,
                        "provider": self._provider,
                        "model": getattr(self.lm, "model", None),
                    }
                    return p

                # If no tool and no final, gently nudge by adding a malformed marker.
                if not executed_any:
                    state.append({"tool": "⛔️no_action", "args": {}, "observation": "No tool_calls or final returned."})
                    # Continue loop

            # Fallback compose from tools
            calculators = [s for s in state if s.get("tool") == "calculator" and isinstance(s.get("observation"), dict)]
            nows = [s for s in state if s.get("tool") == "now" and isinstance(s.get("observation"), dict)]
            parts = []
            if calculators:
                parts.append(str(calculators[0]["observation"].get("result")))
            if nows:
                iso = nows[-1]["observation"].get("iso")
                if iso:
                    parts.append(f"UTC: {iso}")
            p = dspy.Prediction(answer=" | ".join([p for p in parts if p]), trace=state)
            p.usage = {
                "lm_calls": lm_calls,
                "tool_calls": tool_calls,
                "provider": self._provider,
                "model": getattr(self.lm, "model", None),
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
        }
        return p
