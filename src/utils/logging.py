"""Minimal, consistent logging for AEGIS.

Library modules call :func:`get_logger`; entry points (CLI scripts, Streamlit)
call :func:`configure_logging` once.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def configure_logging(level: int | str = logging.INFO) -> None:
    global _configured
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger("aegis")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        configure_logging()
    return logging.getLogger(f"aegis.{name}")
