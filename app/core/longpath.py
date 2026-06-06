r"""Windows long-path (>260 char) safe filesystem writes.

Derived artifacts are written to deeply-nested paths
(_rerun/<uuid>/deals/<uuid>/artifacts/<64-char-sha>/<long name>.derived/
structured.json). On Windows the legacy MAX_PATH=260 limit makes the write
throw FileNotFoundError [WinError 206] AFTER the parser has already done all
the work — silently dropping the whole file's output (empty .derived dir).
This was the dominant cause of "lost files" in local/training runs.

The fix: prefix the Windows extended-length namespace (\\?\) on an absolute,
normalized path, which lifts the 260 limit. No-op on POSIX (4096 limit). Pure
stdlib, never changes behavior on non-Windows.
"""
from __future__ import annotations

import os
from pathlib import Path

_WIN = os.name == "nt"


def long_path(path: os.PathLike | str) -> str:
    """Absolute path string safe for >260-char writes on Windows."""
    s = os.path.abspath(str(path))
    if not _WIN or s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):  # UNC share -> \\?\UNC\server\share\...
        return "\\\\?\\UNC\\" + s.lstrip("\\")
    return "\\\\?\\" + s


def long_mkdir(path: os.PathLike | str) -> None:
    os.makedirs(long_path(path), exist_ok=True)


def long_write_text(path: os.PathLike | str, text: str, *, encoding: str = "utf-8") -> None:
    """mkdir parents + write text, both long-path-safe. Raises on real I/O
    errors (never silently drops) so the caller can surface them."""
    long_mkdir(Path(path).parent)
    with open(long_path(path), "w", encoding=encoding, newline="") as fh:
        fh.write(text)


def long_write_bytes(path: os.PathLike | str, data: bytes) -> None:
    long_mkdir(Path(path).parent)
    with open(long_path(path), "wb") as fh:
        fh.write(data)


__all__ = ["long_path", "long_mkdir", "long_write_text", "long_write_bytes"]
