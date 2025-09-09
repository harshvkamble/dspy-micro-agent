from __future__ import annotations
import logging
import os


def setup_logging() -> None:
    """Configure basic logging for CLI/server usage.

    Controlled via env var MICRO_AGENT_LOG (debug|info|warning|error).
    Defaults to INFO. No-op if already configured.
    """
    if logging.getLogger().handlers:
        return
    level_str = os.getenv("MICRO_AGENT_LOG", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)

