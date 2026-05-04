"""Project-level configuration schema (PRODUCTION_GAPS.md P3.1).

A ``project.yaml`` declares per-project compile defaults so operators
don't have to remember CLI flags or trust filename heuristics.  The
file lives at ``<project>/project.yaml`` and is read by
``app.domain.pack_router.auto_route_pack`` via
``_route_from_project_yaml``.

Schema (all keys optional)::

    # Pinned domain pack — overrides every other auto-routing signal
    domain_pack: security_camera_pack

    # Service line — looked up against the pack synonym table when
    # ``domain_pack`` is absent.  Free text is OK ("video surveillance",
    # "audio visual", "mass notification", etc.).
    service_line: security_camera

    # Free-text project context shown in the review-folder header.
    # Useful for capturing "this RFP includes Q&A from a pre-proposal
    # conference; treat blue-text answers as customer_current_authored".
    context_notes: |
      Virginia Tech Video Surveillance RFP #0016531 Addendum #2.
      Color-coded Q&A — blue answers are customer_current_authored.

    # Optional metadata mirrored into the manifest.
    customer: virginia_tech
    project_name: VT Video Surveillance Addendum 2

    # Optional list of artifact globs to exclude on top of the
    # built-in ``.parserignore`` patterns.
    parserignore_extra:
      - "Attachment D - Sample Agreement.*"
      - "*.draft.pdf"

This module owns the schema; ``pack_router`` reads it.  Tests cover
the round-trip and validation behavior in
``tests/test_project_config.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    domain_pack: str | None = None
    service_line: str | None = None
    context_notes: str | None = None
    customer: str | None = None
    project_name: str | None = None
    parserignore_extra: list[str] = Field(default_factory=list)


def load_project_config(project_dir: Path) -> ProjectConfig | None:
    """Load ``<project>/project.yaml`` (or ``project.yml``) if present.

    Returns ``None`` when no file exists, an empty config when the file
    is empty, or a populated :class:`ProjectConfig` otherwise.  Raises
    ``ValueError`` only on a structurally-bad file (a list at top
    level, syntactically broken YAML).  Unknown keys are ignored so
    operators can scribble notes without breaking the compile.
    """
    for name in ("project.yaml", "project.yml"):
        path = project_dir / name
        if not path.is_file():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML in {path}: {exc}"
            ) from exc
        if payload is None:
            return ProjectConfig()
        if not isinstance(payload, dict):
            raise ValueError(
                f"{path} must be a YAML mapping at the top level (got {type(payload).__name__})"
            )
        # Drop unknown keys so future schema additions don't break old
        # configs and so typos don't crash a compile.  Kept as a list
        # of warnings the caller may surface if desired.
        known = set(ProjectConfig.model_fields)
        cleaned: dict[str, Any] = {k: v for k, v in payload.items() if k in known}
        return ProjectConfig.model_validate(cleaned)
    return None


_DEFAULT_TEMPLATE = """\
# parser-os project configuration
#
# All keys are optional.  When this file is missing or empty,
# parser-os falls back to SOURCE_NOTES.md → filename keywords →
# content scoring → default_pack auto-routing.
#
# Domain pack — pin a specific pack to override all auto-routing.
# Available: security_camera, access_control, wireless, networking,
#            copper_cabling, av, bms, paging, fire_safety, das,
#            electrical, itad, default_pack
# domain_pack: security_camera_pack

# Service line — looked up against the synonym table to pick a pack.
# Free text accepted ("video surveillance", "audio visual", etc.).
# service_line: security_camera

# Free-text project context shown in the review-folder header.
# context_notes: |
#   Brief description of the project, customer authoring conventions,
#   etc.  This is purely human-readable — no parser logic depends on it.

# Optional metadata mirrored into the manifest.
# customer: <slug>
# project_name: <human-readable name>

# Extra glob patterns to ignore on top of the built-in skip list
# (labels/, .orbitbrief/, gold_standard.*, SOURCE_NOTES.md, …).
# parserignore_extra:
#   - "*.draft.pdf"
#   - "vendor_redacted_*.pdf"
"""


def write_default_project_yaml(project_dir: Path) -> Path:
    """Scaffold a ``project.yaml`` template into ``project_dir``.

    Used by ``app.cli init``.  Skips when a config already exists.
    """
    target = project_dir / "project.yaml"
    if target.is_file():
        return target
    target.write_text(_DEFAULT_TEMPLATE, encoding="utf-8")
    return target


__all__ = ["ProjectConfig", "load_project_config", "write_default_project_yaml"]
