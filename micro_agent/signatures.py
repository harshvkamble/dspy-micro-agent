from __future__ import annotations
import dspy
from dspy.adapters import Tool as DSpyTool, ToolCalls

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


class PlanWithTools(dspy.Signature):
    """OpenAI-native tool calling: predict tool calls or finalize.

    - tools (input): a list of dspy.Tool objects with names/args/desc.
    - tool_calls (output): model-selected function calls with arguments.
    - final (output): a final natural language answer (when no more tools needed).
    """
    question: str = dspy.InputField()
    state: str = dspy.InputField(desc="JSON list of prior steps [{tool,args,observation}]")
    tools: list[DSpyTool] = dspy.InputField(desc="Available tools for function calling")
    tool_calls: ToolCalls = dspy.OutputField()
    final: str = dspy.OutputField()
