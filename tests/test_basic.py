from micro_agent.config import configure_lm
from micro_agent.agent import MicroAgent

def test_addition_and_time():
    configure_lm()
    agent = MicroAgent(max_steps=4)
    q = "What's 2*(3+5), then say 'done'."
    pred = agent(q)
    assert "16" in pred.answer

