from __future__ import annotations
import os
from typing import Tuple

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

def get_prices_per_1k(model: str, provider: str) -> Tuple[float, float]:
    # Allow env overrides; default to 0 to avoid misleading values.
    in_price = float(os.getenv("OPENAI_INPUT_PRICE_PER_1K", "0") or 0)
    out_price = float(os.getenv("OPENAI_OUTPUT_PRICE_PER_1K", "0") or 0)
    if provider != "openai":
        return 0.0, 0.0
    return in_price, out_price

def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str, provider: str) -> float:
    in_price_1k, out_price_1k = get_prices_per_1k(model, provider)
    return (input_tokens / 1000.0) * in_price_1k + (output_tokens / 1000.0) * out_price_1k

