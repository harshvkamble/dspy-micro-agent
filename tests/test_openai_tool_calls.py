import os
import pytest

from micro_agent.config import configure_lm
from micro_agent.agent import MicroAgent


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY to run OpenAI tool-calls test",
)
def test_openai_tool_calls_math():
    os.environ["LLM_PROVIDER"] = "openai"
    configure_lm()
    agent = MicroAgent(max_steps=4)
    q = "What's 2*(3+5)? Return only the number."
    pred = agent(q)
    assert "16" in pred.answer
    assert any(step.get("tool") == "calculator" for step in pred.trace)

