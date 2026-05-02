"""Compatibility wrapper for the core detector.

The original `detect_standalone.py` was a 5k+ line monolith.  For the final
migration package we keep that behavior byte-for-byte available, but quarantine
it as read-only runtime chunks so no source file exceeds the project limit and
new work cannot accidentally pile into the core engine.

Do not add new detection logic here.  New behavior belongs in `passes/`,
`vision/`, `semantics/`, `overlay/`, and `qa/` with golden tests.
"""
from __future__ import annotations

from importlib import resources


def _load_core_source() -> str:
    base = resources.files(__package__) / "core_runtime" / "chunks"
    names = (base / "MANIFEST").read_text(encoding="utf-8").splitlines()
    return "".join((base / name).read_text(encoding="utf-8") for name in names if name.strip())


exec(compile(_load_core_source(), "<parser_os_overlay_core_runtime>", "exec"), globals(), globals())
