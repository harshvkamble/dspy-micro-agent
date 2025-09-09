from __future__ import annotations
import os
import dspy
from dotenv import load_dotenv

load_dotenv()

def configure_lm():
    """
    Configure DSPy's LM with multiple provider fallbacks.

    Priority order (controlled by env):
    - If LLM_PROVIDER=ollama (or OLLAMA_MODEL is set), try Ollama first.
    - Else default to OpenAI (OPENAI_MODEL).

    Env vars:
    - LLM_PROVIDER: 'ollama' | 'openai' (default: 'openai')
    - OPENAI_MODEL, OPENAI_API_KEY
    - OLLAMA_MODEL (e.g., 'llama3.2:1b', 'llama3.1', 'qwen2.5')
    - OLLAMA_HOST (default 'http://localhost:11434')
    - TEMPERATURE, MAX_TOKENS
    """
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    temperature = float(os.getenv("TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("MAX_TOKENS", "1024"))

    tried = []

    # Helper to try multiple backends safely
    def _try(name, fn):
        try:
            lm = fn()
            dspy.settings.configure(lm=lm)
            return True
        except Exception as e:
            tried.append((name, repr(e)))
            return False

    # Option 1: Ollama (explicit or inferred via OLLAMA_MODEL)
    if provider == "ollama" or os.getenv("OLLAMA_MODEL"):
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        # Try explicit dspy Ollama wrapper (if available in this DSPy version)
        if _try("dspy.Ollama", lambda: dspy.Ollama(model=ollama_model, base_url=ollama_host, temperature=temperature, max_tokens=max_tokens)):
            return
        # Try generic registry form
        if _try("dspy.LM(ollama/<model>)", lambda: dspy.LM(f"ollama/{ollama_model}")):
            return

    # Option 2: OpenAI (default)
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if _try("dspy.OpenAI", lambda: dspy.OpenAI(model=openai_model, temperature=temperature, max_tokens=max_tokens)):
        return
    if _try("dspy.LM(openai/<model>)", lambda: dspy.LM(f"openai/{openai_model}")):
        return

    detail = "\n".join([f"- {name}: {err}" for name, err in tried])
    raise RuntimeError(
        "Failed to configure DSPy LM. Provide either an OPENAI_API_KEY or a running Ollama with models.\n"
        + detail
    )
