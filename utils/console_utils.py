"""
Console helpers for Windows-safe logging.
"""

from __future__ import annotations

import builtins
import sys
from typing import Any


_ORIGINAL_PRINT = builtins.print
_PATCHED = False


def _safe_text(value: Any, encoding: str) -> str:
    text = str(value)
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def safe_print(*args: Any, **kwargs: Any) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_args = tuple(_safe_text(arg, encoding) for arg in args)
    _ORIGINAL_PRINT(*safe_args, **kwargs)


def setup_console() -> None:
    global _PATCHED
    if _PATCHED:
        return

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                try:
                    stream.reconfigure(errors="replace")
                except Exception:
                    pass

    builtins.print = safe_print
    _PATCHED = True
