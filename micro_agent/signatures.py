from __future__ import annotations
import dspy

class PlanOrAct(dspy.Signature):
    """Decide next step: either call a tool with JSON args or finalize.

Rules:
- Always use the 'calculator' tool for any arithmetic beyond trivial mental math.
- Always use the 'now' tool for current time/date questions. Do not guess.
- You may call at most one tool per step.
- Respond with STRICT JSON only; no extra text or code fences.
    """
    question: str = dspy.InputField()
    state: str = dspy.InputField(desc="JSON list of prior steps [{tool,args,observation}]")
    tools: str = dspy.InputField(desc="JSON list of available tools with name, description, schema")
    decision: str = dspy.OutputField(desc="""
Return STRICT JSON in one of these shapes. Use tools when applicable:

{"tool": {"name": "calculator", "args": {"expression": "..."}}}
{"tool": {"name": "now", "args": {"timezone": "utc"}}}

OR finalize only when all info is gathered:
{"final": {"answer": "<complete answer string>"}}
""")

class Finalize(dspy.Signature):
    """Given the question and full trace, produce the final answer."""
    question: str = dspy.InputField()
    state: str = dspy.InputField(desc="JSON list of prior steps")
    answer: str = dspy.OutputField()
