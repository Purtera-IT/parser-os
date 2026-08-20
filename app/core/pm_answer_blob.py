"""Load the PM's answered questions from blob so they enter the compile as evidence.

The brief writes every teach action to ``question_feedback.jsonl``. The
``answered`` rows are the valuable ones: a PM typing what the customer actually
confirmed is usually deal truth that appears in no document. Reading them here —
BEFORE graph_build and packetize — makes them behave like any other evidence:
they get edges, land in packets, can settle a cross-document conflict, and reach
the SOW. See :mod:`app.core.pm_answer_atoms` for the atom shape.

Contract mirrors :mod:`app.core.feedback_blob`:

* **Gated**: no-op unless ``SOWSMITH_FEEDBACK_BLOB`` is truthy.
* **Offline-safe**: any failure (missing dep, no conn string, network, bad JSON)
  yields an empty list — a missing ledger must never break a compile.
* **Deterministic**: atom ids hash from (project, question, answer), so
  recompiling the same deal produces the same atoms rather than duplicates.

Layout (written by the Platform-infra question-feedback function):
``<container>/deals/<deal_id>/orbitbrief/latest/question_feedback.jsonl``
"""
from __future__ import annotations

import json
import os

from app.core.env import env_get
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}


def _enabled() -> bool:
    return env_get("PARSER_OS_FEEDBACK_BLOB", "").strip().lower() in _TRUTHY


def _container_client():
    """A blob ContainerClient, or ``None`` when disabled/unconfigured/offline."""
    if not _enabled():
        return None
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if not conn:
        return None
    try:
        from azure.storage.blob import BlobServiceClient
    except Exception:
        return None
    container = env_get("PARSER_OS_FEEDBACK_BLOB_CONTAINER", "orbitbrief-artifacts"
    ).strip() or "orbitbrief-artifacts"
    try:
        svc = BlobServiceClient.from_connection_string(conn)
        return svc.get_container_client(container)
    except Exception:
        return None


def parse_ledger(text: str) -> list[dict[str, Any]]:
    """Parse JSONL into rows, skipping malformed lines.

    Tolerant on purpose: one truncated line must not cost us every answer the
    PM has given on this deal.
    """
    rows: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_answer_events(deal_id: str) -> list[dict[str, Any]]:
    """Fetch this deal's feedback ledger rows from blob. ``[]`` when unavailable."""
    if not deal_id:
        return []
    cc = _container_client()
    if cc is None:
        return []
    blob = f"deals/{deal_id}/orbitbrief/latest/question_feedback.jsonl"
    try:
        raw = cc.download_blob(blob).readall()
    except Exception:
        # No teaching on this deal yet, or blob unreachable — both are normal.
        return []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return parse_ledger(raw)


def load_pm_answer_atoms(*, project_id: str, deal_id: str = "") -> list[Any]:
    """The deal's answered questions as :class:`EvidenceAtom` rows.

    Returns ``[]`` on any failure — a compile must never depend on the ledger
    being reachable.
    """
    try:
        events = load_answer_events(deal_id or project_id)
        if not events:
            return []
        from app.core.pm_answer_atoms import pm_answers_to_atoms

        return pm_answers_to_atoms(events, project_id=project_id)
    except Exception:
        return []
