"""Environment lookup that accepts both the new and the legacy prefix.

parser-os was originally called SowSmith, and 128 environment variables still
carry that name -- ``SOWSMITH_DISABLE_LLM``, ``SOWSMITH_FEEDBACK_BLOB``,
``SOWSMITH_ML_PROFILE``. The name has since been taken by a real and separate
component: ``Purtera-IT/SowSmith``, the deterministic Statement of Work
generator that Core imports as ``from sowsmith import build_sow_markdown``.

So one name now denotes two unrelated things, and the collision is not
cosmetic: someone debugging ``SOWSMITH_FEEDBACK_BLOB`` would reasonably open
the SowSmith repository, where nothing of the sort exists.

Renaming cannot be a single commit. Those variables are set in every container
app, job and function app -- 56 on the dev parser worker alone -- so flipping
the code and the deployments together means one atomic change across every
environment, and any variable missed silently reverts a behaviour flag rather
than failing loudly.

Hence dual-read. ``env_get("PARSER_OS_X")`` tries ``PARSER_OS_X`` first and
falls back to ``SOWSMITH_X``. Today no ``PARSER_OS_*`` variable is set
anywhere, so every lookup resolves through the fallback and behaviour is
identical. Deployments can then be migrated one at a time, in any order, with
both names live throughout; the fallback is removed only once nothing sets the
old prefix.
"""

from __future__ import annotations

import os

_NEW = "PARSER_OS_"
_OLD = "SOWSMITH_"

_MISSING = object()


def _both(name: str) -> tuple[str, str]:
    """The new and legacy spellings of one variable, given either."""
    bare = name
    for p in (_NEW, _OLD):
        if bare.startswith(p):
            bare = bare[len(p):]
            break
    return _NEW + bare, _OLD + bare


def env_get(name: str, default: str | None = None) -> str | None:
    """``os.environ.get`` over both prefixes, new name winning.

    Accepts the variable written either way, so a call site can be migrated to
    the new spelling without waiting for the deployments.
    """
    new, old = _both(name)
    v = os.environ.get(new, _MISSING)  # type: ignore[arg-type]
    if v is not _MISSING:
        return v  # type: ignore[return-value]
    return os.environ.get(old, default)


def env_flag(name: str, default: bool = False) -> bool:
    """A truthy switch, read over both prefixes.

    Centralised because the codebase spells this test at least four different
    ways -- ``in ("1","true","yes","on")``, ``.lower() == "1"``,
    ``in _TRUTHY`` -- and they disagree at the edges.
    """
    v = env_get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def env_set_legacy_aliases() -> int:
    """Mirror every ``SOWSMITH_*`` into ``PARSER_OS_*`` in this process.

    For subprocesses and third-party code that read ``os.environ`` directly and
    cannot be routed through :func:`env_get`. Returns how many were mirrored.
    """
    n = 0
    for k, v in list(os.environ.items()):
        if k.startswith(_OLD):
            new = _NEW + k[len(_OLD):]
            if new not in os.environ:
                os.environ[new] = v
                n += 1
    return n
