"""PurTera's SOW standards: the packaged file plus items added without a deploy.

``app/core/purtera_standards.json`` holds PurTera's fixed handling per job
family (method steps, customer duties, assumptions, exclusions, deliverables,
tools, materials). Template gaps were the biggest cause in the pass Z grades
(~35 tagged items vs ~4 head errors), and every new item needed a code
deploy. This module merges an OVERLAY on top of the packaged file:

    {"items": [{"id", "family", "section", "text", "when"?, "unless"?,
                "status": "active" | "retired", "added_by", "added_at",
                "source_deal"?, "note"?}]}

An active overlay item is appended to its family/section (``{when:<regex>}``
/ ``{unless:<regex>}`` prefixes as the packaged file writes them); a retired
item whose text matches a packaged item removes that item. The overlay lives
in blob storage (``SOWSMITH_STANDARDS_CONTAINER``, default
``orbitbrief-artifacts``, name ``_sow_standards/overlay.json``) or, for tests
and local runs, ``SOWSMITH_STANDARDS_OVERLAY_FILE``. It is read once per
``SOWSMITH_STANDARDS_TTL_S`` (300 s) per process; any failure falls back to
the packaged file alone. Writes use the blob ETag, so two editors never
overwrite each other.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PACKAGED = Path(__file__).with_name("purtera_standards.json")
BLOB_NAME = "_sow_standards/overlay.json"
SECTIONS = ("method_steps", "customer_responsibilities", "assumptions", "exclusions", "deliverable_outcomes",
            "tools", "materials")

_lock = threading.Lock()
_cache: dict[str, Any] = {"at": 0.0, "overlay": None}


class ConflictError(RuntimeError):
    """Someone else changed the overlay since it was read."""


def _ttl() -> float:
    try:
        return float(os.environ.get("SOWSMITH_STANDARDS_TTL_S", "300"))
    except ValueError:
        return 300.0


def _file() -> str:
    return os.environ.get("SOWSMITH_STANDARDS_OVERLAY_FILE", "").strip()


def _container():
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if not conn:
        return None
    from azure.storage.blob import BlobServiceClient

    name = os.environ.get("SOWSMITH_STANDARDS_CONTAINER", "orbitbrief-artifacts")
    return BlobServiceClient.from_connection_string(conn).get_container_client(name)


def read_overlay() -> tuple[dict[str, Any], str | None]:
    """(overlay, etag). Empty overlay when none is stored or storage is unreachable."""
    path = _file()
    if path:
        p = Path(path)
        if not p.exists():
            return {"items": []}, None
        data = json.loads(p.read_text())
        return data, str(p.stat().st_mtime_ns)
    try:
        c = _container()
        if c is None:
            return {"items": []}, None
        b = c.get_blob_client(BLOB_NAME)
        dl = b.download_blob()
        return json.loads(dl.readall()), dl.properties.etag
    except Exception as e:  # missing blob, no network: packaged only
        if "BlobNotFound" not in str(e):
            log.info("standards overlay unreadable (%s); packaged standards only", type(e).__name__)
        return {"items": []}, None


def write_overlay(data: dict[str, Any], etag: str | None) -> None:
    payload = json.dumps(data, indent=1, ensure_ascii=False)
    path = _file()
    if path:
        p = Path(path)
        if etag is not None and p.exists() and str(p.stat().st_mtime_ns) != etag:
            raise ConflictError("overlay changed since it was read")
        p.write_text(payload)
    else:
        c = _container()
        if c is None:
            raise RuntimeError("no storage configured for the standards overlay")
        from azure.core import MatchConditions

        kw = {"etag": etag, "match_condition": MatchConditions.IfNotModified} if etag else {}
        try:
            c.get_blob_client(BLOB_NAME).upload_blob(payload, overwrite=True, **kw)
        except Exception as e:
            if "ConditionNotMet" in str(e) or "412" in str(e):
                raise ConflictError("overlay changed since it was read") from e
            raise
    with _lock:
        _cache["at"] = 0.0


def _overlay_cached() -> dict[str, Any]:
    with _lock:
        if _cache["overlay"] is not None and time.monotonic() - _cache["at"] < _ttl():
            return _cache["overlay"]
    ov, _ = read_overlay()
    with _lock:
        _cache.update(overlay=ov, at=time.monotonic())
    return ov


def _strip(t: str) -> str:
    return re.sub(r"^(\{(?:when|unless):.*?\}\s+)+(?=[A-Z])", "", str(t)).strip().lower()


def item_text(it: dict[str, Any]) -> str:
    """The item as the packaged file writes it, gates first."""
    t = str(it.get("text") or "").strip()
    if it.get("when"):
        t = "{when:%s} %s" % (it["when"], t)
    if it.get("unless"):
        t = "{unless:%s} %s" % (it["unless"], t)
    return t


def merge(packaged: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Packaged standards with the overlay applied. Pure."""
    out = json.loads(json.dumps(packaged))
    for it in overlay.get("items") or []:
        fam, sec = str(it.get("family") or ""), str(it.get("section") or "")
        if not fam or sec not in SECTIONS or not str(it.get("text") or "").strip():
            continue
        block = out.setdefault(fam, {})
        lst = block.setdefault(sec, [])
        if it.get("status", "active") == "retired":
            want = _strip(str(it.get("text")))
            block[sec] = [t for t in lst if _strip(t) != want]
        elif _strip(item_text(it)) not in {_strip(t) for t in lst}:
            lst.append(item_text(it))
    return out


def load() -> dict[str, Any]:
    """Every family's standards: packaged + overlay (packaged only on any failure)."""
    try:
        packaged = json.loads(PACKAGED.read_text())
    except Exception:
        return {}
    try:
        return merge(packaged, _overlay_cached())
    except Exception:
        log.exception("standards overlay merge failed; packaged standards only")
        return packaged


def _validate(it: dict[str, Any]) -> None:
    if it.get("section") not in SECTIONS:
        raise ValueError(f"section must be one of {', '.join(SECTIONS)}")
    if not re.match(r"^[a-z0-9_]+$", str(it.get("family") or "")):
        raise ValueError("family must be a job family key like av_conference_room")
    text = re.sub(r"^(\{(?:when|unless):.*?\}\s+)+", "", str(it.get("text") or "")).strip()
    if len(text) < 8 or not text[0].isupper():
        raise ValueError("text must be a full sentence starting with a capital letter")
    for k in ("when", "unless"):
        if it.get(k):
            re.compile(str(it[k]))  # raises re.error on a bad pattern


def add_item(*, family: str, section: str, text: str, added_by: str, when: str = "", unless: str = "",
             source_deal: str = "", note: str = "") -> dict[str, Any]:
    it = {"id": "std_" + uuid.uuid4().hex[:12], "family": family, "section": section, "text": text.strip(),
          "status": "active", "added_by": added_by, "added_at": datetime.now(timezone.utc).isoformat(),
          "source_deal": source_deal, "note": note}
    if when:
        it["when"] = when
    if unless:
        it["unless"] = unless
    _validate(it)
    ov, etag = read_overlay()
    ov.setdefault("items", []).append(it)
    write_overlay(ov, etag)
    return it


def retire_item(item_id: str = "", *, family: str = "", section: str = "", text: str = "", by: str = "") -> dict[str, Any]:
    """Retire an overlay item by id, or a packaged item by family/section/text."""
    ov, etag = read_overlay()
    for it in ov.get("items") or []:
        if item_id and it.get("id") == item_id:
            it.update(status="retired", retired_by=by, retired_at=datetime.now(timezone.utc).isoformat())
            write_overlay(ov, etag)
            return it
    if item_id:
        raise KeyError(item_id)
    it = {"id": "std_" + uuid.uuid4().hex[:12], "family": family, "section": section, "text": text,
          "status": "retired", "added_by": by, "added_at": datetime.now(timezone.utc).isoformat()}
    _validate(it)
    ov.setdefault("items", []).append(it)
    write_overlay(ov, etag)
    return it


__all__ = ["load", "merge", "read_overlay", "write_overlay", "add_item", "retire_item", "item_text",
           "SECTIONS", "ConflictError"]
