from __future__ import annotations
import os
from typing import Tuple, Any, Dict

def _try_tiktoken(model: str):
    try:
        import tiktoken
        # Use a generic encoding if specific not found
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("o200k_base")
        return enc
    except Exception:
        return None

def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    if not text:
        return 0
    enc = _try_tiktoken(model)
    if enc is None:
        # Fallback heuristic: ~4 chars per token
        return max(1, len(text) // 4)
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)

_OPENAI_DEFAULTS = {
    # Prices per 1K tokens (USD) — see https://platform.openai.com/docs/pricing
    # These defaults are best-effort and may drift. Override via env vars to be exact.
    "gpt-4o-mini": (0.00015, 0.0006),  # $0.15 / $0.60 per 1M
    "gpt-4o": (0.005, 0.015),          # $5 / $15 per 1M
    "gpt-4.1": (0.005, 0.015),         # typical parity with 4o
}

def _normalize(model: str) -> str:
    if not model:
        return ""
    m = model.lower()
    # Strip provider prefix like 'openai/' if present
    if "/" in m:
        m = m.split("/", 1)[1]
    return m

def get_prices_per_1k(model: str, provider: str) -> Tuple[float, float]:
    # Allow env overrides; when set, they win.
    env_in = os.getenv("OPENAI_INPUT_PRICE_PER_1K")
    env_out = os.getenv("OPENAI_OUTPUT_PRICE_PER_1K")
    if provider == "openai":
        if env_in is not None and env_out is not None:
            try:
                return float(env_in or 0), float(env_out or 0)
            except Exception:
                pass
        key = _normalize(model)
        # Match by prefix to handle variants like gpt-4o-mini-2024-xx-xx
        for base, prices in _OPENAI_DEFAULTS.items():
            if key.startswith(base):
                return prices
    return 0.0, 0.0

def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str, provider: str) -> float:
    in_price_1k, out_price_1k = get_prices_per_1k(model, provider)
    return (input_tokens / 1000.0) * in_price_1k + (output_tokens / 1000.0) * out_price_1k

def estimate_prediction_cost(question: str, trace: Any, answer: str, usage: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate token usage and USD cost for a single prediction.

    Heuristic: input tokens ~= lm_calls * tokens(question) + tokens(str(trace))
               output tokens ~= tokens(answer)
    """
    provider = (usage or {}).get("provider") or "openai"
    model = (usage or {}).get("model") or "gpt-4o-mini"
    lm_calls = int((usage or {}).get("lm_calls", 0) or 0)

    q_tokens = estimate_tokens(str(question or ""), model)
    trace_tokens = estimate_tokens(str(trace or ""), model)
    ans_tokens = estimate_tokens(str(answer or ""), model)
    in_tokens = lm_calls * q_tokens + trace_tokens
    out_tokens = ans_tokens
    cost = estimate_cost_usd(in_tokens, out_tokens, model=model, provider=provider)
    return {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "cost_usd": cost,
    }
