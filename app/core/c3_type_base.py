"""The C3 base as a deflect layer on the atom-type stage.

A trained C3 checkpoint (``ml/c3``) reads the whole deal at once -- every
line in context, in arrival order -- and types each line. Where it is sure,
the typed atom classifier can take its answer and skip the LLM for that atom.
Same contract as the other deflect layers: one-sided, guess-free, OFF by
default.

    SOWSMITH_C3_TYPE         off (default) | 1/true/yes/on | shadow
    SOWSMITH_C3_CHECKPOINT   path to the trained checkpoint
    SOWSMITH_C3_ENCODER      the line encoder the checkpoint was trained on
                             (a hub name or a local path)
    SOWSMITH_C3_SURE         probability a reading must reach to act (0.7)
    SOWSMITH_C3_DEVICE       cuda | mps | cpu (default: the first present)
    SOWSMITH_C3_TYPES        the answers it may act on (default: every
                             value-light type and ``_keep``)
    SOWSMITH_C3_DTYPE        fp32 (default) | bf16: the layers above the
                             encoder in bf16, about half their memory
                             (the encoder is bf16 either way)
    SOWSMITH_C3_MIN_FREE_MB  memory the container must have free before the
                             model loads (2500); short of it, the layer
                             abstains instead of risking the compile
    SOWSMITH_C3_BUDGET_S     longest a deal read may take (90); past it the
                             read stops and every atom goes to the LLM
    SOWSMITH_C3_URL          a base service (``ml.c3.serve``, on a GPU) to
                             read the deal instead of loading the model here;
                             the worker then needs no model memory. Past the
                             budget or on any error, every atom goes to the LLM
    SOWSMITH_C3_TOKEN        the service's bearer token, when it has one
    SOWSMITH_C3_CONNECT_S    how long the service's /health may take before
                             the read is skipped as unreachable (5)
    SOWSMITH_C3_SHADOW_WAIT_S  longest the stage waits for a shadow read
                             after the LLM is done (20); past it the shadow
                             is dropped, so shadow never holds a compile

``shadow`` reads the deal and reports what the layer would have done, beside
what the compile then did, as one ``c3_type_shadow`` event; it changes
nothing. The model is loaded once per process; a failed load is logged once
and the layer abstains from then on.

The labeling page export is what the model trained on (``ml/c3/export.py``);
``atoms_to_deal`` builds the same rows from the compiler's atoms. Where an
export field has no source on an atom it takes export.py's default, said at
the field.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

log = logging.getLogger(__name__)

_TRUE = ("1", "true", "yes", "on")

#: Readings the layer never acts on: our own pricing workbook and the
#: estimating calc. The live parser does not read them and the base was not
#: scored on them (``ml.c3.loop.CHECK_SKIP_DOC_KINDS``).
_SKIP_KINDS = ("deal_kit", "estimating_calc")

#: The window the base was trained and scored with (ml/c3/pod/*.sh ATOMS):
#: a deal past this is read in windows. Reading smaller windows than it saw
#: in training changes its answers.
MAX_ATOMS = 8192


# What the layer last did, for the compile's ML manifest and the nightly
# report: state is off | loaded | load_failed | no_memory | over_budget |
# warming | read_failed | read. Counts and times only.
_STATUS: dict[str, Any] = {"state": "off"}


def status() -> dict[str, Any]:
    return dict(_STATUS)


def note_typing(why: dict[str, Any]) -> None:
    """What the typing stage did with the base's readings (``on``): of the
    lines it would send the LLM, how many the base took, and why the rest were
    not its -- no reading, unsure, a skipped kind, or a sure answer the list
    does not take (by answer). Lands in the compile's ML manifest beside the
    read's own status. Counts only."""
    _STATUS["typing"] = dict(why)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _free_mb() -> float | None:
    """Memory this container can still use, from its cgroup (v2, then v1);
    None when no limit is visible (a laptop, an unlimited box)."""
    for limit_p, used_p in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                            ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                             "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            with open(limit_p) as f:
                raw = f.read().strip()
            if raw == "max":
                return None
            limit = int(raw)
            if limit >= 1 << 60:
                return None
            with open(used_p) as f:
                used = int(f.read().strip())
            return (limit - used) / 1e6
        except (OSError, ValueError):
            continue
    return None


def mode() -> str:
    """``off``, ``shadow`` or ``on``."""
    v = os.environ.get("SOWSMITH_C3_TYPE", "").strip().lower()
    if v == "shadow":
        return "shadow"
    return "on" if v in _TRUE else "off"


def enabled() -> bool:
    return mode() != "off"


def sure() -> float:
    try:
        return float(os.environ.get("SOWSMITH_C3_SURE", "0.7"))
    except ValueError:
        return 0.7


def sure_for(answer: str) -> float:
    """The certainty an answer must reach to act: ``SOWSMITH_C3_SURE_BY_TYPE``
    (``type:p,...``) where set, else ``SOWSMITH_C3_SURE``. Each type's bar is
    the lowest band where the base matches or beats the parser on the same
    lines (validation deal 010043, 2026-10-15)."""
    raw = os.environ.get("SOWSMITH_C3_SURE_BY_TYPE", "")
    for part in raw.split(","):
        name, _, val = part.strip().partition(":")
        if name.strip() == answer:
            try:
                return float(val)
            except ValueError:
                break
    return sure()


def acts_on() -> frozenset[str] | None:
    """The answers the layer may act on (``SOWSMITH_C3_TYPES``, comma
    separated, ``_keep`` and ``small_talk`` included by name), or None when unset. Unset, ``on``
    takes nothing and ``shadow`` reports every value-light type and ``_keep``. Promotion widens this one type at a time, each only
    after it clears the route report (``ml.c3.route_report``); shadow reports
    exactly what ``on`` would do with the same list."""
    raw = os.environ.get("SOWSMITH_C3_TYPES", "").strip()
    if not raw:
        return None
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


# --------------------------------------------------------------- the model

_LOCK = threading.Lock()
_MODEL: Any = None
_TRIED = False


def _device() -> str:
    import torch

    dev = os.environ.get("SOWSMITH_C3_DEVICE", "").strip().lower()
    if dev and dev != "auto":
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load() -> tuple[Any, Any, str]:
    """(model, schema, device), as predict.py builds and loads them."""
    from dataclasses import asdict

    from ml.c3.gates import load_checkpoint
    from ml.c3.loop import live_student
    from ml.c3.readiness import digest
    from ml.c3.schema import load_schema

    path = os.environ.get("SOWSMITH_C3_CHECKPOINT", "").strip()
    encoder = os.environ.get("SOWSMITH_C3_ENCODER", "").strip()
    if not path or not os.path.exists(path):
        raise FileNotFoundError("SOWSMITH_C3_CHECKPOINT is unset or missing")
    if not encoder:
        raise ValueError("SOWSMITH_C3_ENCODER is unset")
    need = _env_float("SOWSMITH_C3_MIN_FREE_MB", 2500.0)
    free = _free_mb()
    if free is not None and free < need:
        raise MemoryError(f"{free:.0f} MB free, {need:.0f} MB needed")
    import torch

    schema = load_schema()
    device = _device()
    model = live_student(schema, encoder).to(device)
    load_checkpoint(path, model, schema_sha256=digest([asdict(o) for o in schema.opportunities]))
    from ml.c3.loop import to_read_precision

    to_read_precision(model)
    if os.environ.get("SOWSMITH_C3_DTYPE", "").strip().lower() in ("bf16", "bfloat16"):
        model.to(torch.bfloat16)
    model.eval()
    return model, schema, device


def get_model() -> tuple[Any, Any, str] | None:
    """The process-wide model, or None when it cannot load (warned once)."""
    global _MODEL, _TRIED
    if _TRIED:
        return _MODEL
    with _LOCK:
        if not _TRIED:
            started = time.monotonic()
            try:
                _MODEL = _load()
                _STATUS.update(state="loaded", reason="")
            except Exception as exc:
                _MODEL = None
                _STATUS.update(state="no_memory" if isinstance(exc, MemoryError) else "load_failed",
                               reason=f"{type(exc).__name__}: {exc}"[:300])
                log.warning("C3 type base not loaded (%s: %s); the layer abstains",
                            type(exc).__name__, exc)
            _STATUS["load_ms"] = round((time.monotonic() - started) * 1000, 1)
            _TRIED = True
    return _MODEL


# ------------------------------------------------------- atoms -> a deal

def _locator(atom: Any) -> dict[str, Any]:
    try:
        refs = getattr(atom, "source_refs", None) or []
        loc = getattr(refs[0], "locator", None) if refs else None
        return loc if isinstance(loc, dict) else {}
    except Exception:
        return {}


def _ref(atom: Any, name: str) -> str:
    try:
        refs = getattr(atom, "source_refs", None) or []
        v = getattr(refs[0], name, None) if refs else None
        return str(getattr(v, "value", v) or "")
    except Exception:
        return ""


def _value(atom: Any) -> dict[str, Any]:
    v = getattr(atom, "value", None)
    return v if isinstance(v, dict) else {}


#: Every document kind the base trained on (``c3_record.TRAINED_KINDS``):
#: anything else reads as ``file``, so the model never sees a tag it never
#: trained on.
from c3_record import TRAINED_KINDS  # noqa: E402
_EMAIL_TYPES = ("email", "msg", "mbox")


def _doc_id(atom: Any) -> str:
    return str(getattr(atom, "artifact_id", "") or _ref(atom, "artifact_id"))


def _event_kind(atom: Any) -> str:
    """``email``, ``note`` or ``meeting`` when the atom says so, else ''."""
    art = (_ref(atom, "artifact_type") or "").strip().lower()
    if art in _EMAIL_TYPES:
        return "email"
    if str(_value(atom).get("kind") or "").startswith("hubspot_note"):
        return "note"
    if art == "transcript" or "utterance_index" in _locator(atom):
        return "meeting"
    return ""


def _lifecycle(sha: str | None) -> dict[str, Any]:
    """The document lifecycle record for a file's bytes, as the envelope looks
    it up (its ``lifecycle``: type, role, delivered), else {}."""
    if not sha:
        return {}
    try:
        from app.core.document_lifecycle.dataset import lookup

        return lookup(sha) or {}
    except Exception:
        return {}


def _file_kind(atom: Any, artifact_sha256: dict[str, str] | None) -> str:
    """A file's kind by the deal record's rule (``c3_record.file_kind``, the
    one the training export uses): its lifecycle type, looked up by its bytes'
    sha256 as the envelope does (the page's ``docType``); else ``deal_kit`` /
    ``estimating_calc`` / ``site_list`` by the file's name; else ``file``."""
    from c3_record import file_kind

    sha = (artifact_sha256 or {}).get(_doc_id(atom))
    return file_kind(_ref(atom, "filename"), _lifecycle(sha).get("type"))


def doc_kind(atom: Any, artifact_sha256: dict[str, str] | None = None) -> str:
    """The training files' ``doc_kind`` for this atom's document.

    ``artifact_sha256``: artifact id -> the file's sha256 (the compile
    manifest's fingerprints); without it a file falls back to its name."""
    return _event_kind(atom) or _file_kind(atom, artifact_sha256)


def skipped(atom: Any, artifact_sha256: dict[str, str] | None = None) -> bool:
    # c3_record, not ml.c3.data: the worker ships without ml/, and this runs
    # on every sure reading (an ImportError here disabled the whole layer).
    from c3_record import is_deal_kit

    kind = doc_kind(atom, artifact_sha256)
    return kind in _SKIP_KINDS or is_deal_kit(kind)


def _section(atom: Any) -> str:
    """The labeling page's section: the envelope's own ``_atom_section_path``
    (a locator's ``section_path``, else its section, sheet, speaker, channel or
    location, then the email list rules), joined as export.py joins it. A
    transcript line's section is its speaker, as training saw it."""
    try:
        from app.core.orbitbrief_envelope import _atom_section_path

        path = _atom_section_path(atom)
    except Exception:
        path = []
    if path:
        return " > ".join(str(x) for x in path if str(x).strip())
    loc = _locator(atom)
    sp = loc.get("section_path")
    if isinstance(sp, str) and sp.strip():
        return sp
    sheet = loc.get("sheet")
    return str(sheet) if sheet else ""


def _speaker(atom: Any) -> tuple[str, str]:
    """(role, side): the page's ``saidBy``, which is ``value.said_by``. A
    transcript line's locator ``speaker_role`` is not it: training has '' on
    99.98% of meeting lines, and ``unknown`` there meant an email sender."""
    from c3_record import speaker

    by = _value(atom).get("said_by")
    if not isinstance(by, dict):
        return "", ""
    return speaker(by.get("role") or by.get("role_guess"), by.get("side"))


def _sender_inputs(atom: Any) -> dict[str, Any]:
    """What ``c3_record.sender`` reads off one line, as the labeling page
    exports it (atom-labeling.js): ``saidBy`` / ``saidTo`` (``value.said_by``
    / ``said_to``), the email message it was written in (``messageOf``: the
    thread stage's ``email_thread.message.author``, else the locator's
    ``sender``, for a line with a message index), and a call transcript's
    speaker (the locator's ``speaker``, the page's section for that line)."""
    v, loc = _value(atom), _locator(atom)
    out: dict[str, Any] = {}
    by = v.get("said_by")
    if isinstance(by, dict):
        out["said_by"] = {"name": by.get("name"), "email": by.get("email"), "side": by.get("side"),
                          "role": by.get("role") or by.get("role_guess")}
    to = v.get("said_to")
    if isinstance(to, list):
        out["said_to"] = [{"name": p.get("name"), "email": p.get("email")} for p in to if isinstance(p, dict)]
    et = v.get("email_thread") if isinstance(v.get("email_thread"), dict) else {}
    msg = et.get("message") if isinstance(et.get("message"), dict) else {}
    index = msg.get("index") if msg.get("index") is not None else loc.get("message_index", v.get("message_index"))
    if index not in (None, ""):
        author = msg.get("author") or loc.get("sender") or v.get("sender")
        if author:
            out["message_author"] = str(author)
    if (_ref(atom, "artifact_type") or "").strip().lower() == "transcript" or "utterance_index" in loc:
        name = str(loc.get("speaker") or "").strip()
        if name:
            out["speaker_name"] = name
    return out


def _when(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    for parse in (lambda x: datetime.fromisoformat(x.replace("Z", "+00:00")), parsedate_to_datetime):
        try:
            dt = parse(s)
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def _own_time(atom: Any) -> float | None:
    """The atom's own source time: export.py's ``docOrigin.sent_at`` (the
    message a line was written in)."""
    loc = _locator(atom)
    return _when(loc.get("sent_at")) or _when(loc.get("authored_at"))


def _key(atom: Any, i: int) -> str:
    return str(getattr(atom, "id", "") or getattr(atom, "atom_id", "") or f"atom:{i}")


def atoms_to_deal(atoms: list[Any], artifact_sha256: dict[str, str] | None = None):
    """One ``DealExample`` from a compile's atoms (see ``_build``)."""
    return _build(atoms, artifact_sha256)[0]


def _build(atoms: list[Any], artifact_sha256: dict[str, str] | None = None):
    """One ``DealExample`` from a compile's atoms, in arrival order.

    Field by field, against export.py:

    * key / atom_id: the atom's id (the page's ``labelKey`` is a label's key,
      not a model input; any stable key reads the same);
    * text: ``raw_text``;
    * doc_id: ``artifact_id``; doc_kind: ``doc_kind`` above, one per
      document: an artifact any of whose atoms is an email, note or meeting
      line is that kind throughout;
    * section: the locator's ``section_path`` joined with " > ", else its sheet;
    * speaker_side / speaker_role / speaker_org, doc_direction, stage_at:
      ``c3_record.sender`` (``build_rows``), from ``value.said_by``, else the
      line's email message author, else a transcript line's speaker, else its
      document's ``origin`` (``deal_documents``: an attachment's is the
      message that carried it); stage from the deal's CRM stage history when
      it has two or more transitions (``set_deal``'s ``stage_timeline``),
      else from the deal's own milestones;
    * entered_at: the locator's ``sent_at``; else the time of the document
      before it in arrival order (a document's own origin and ``authoredAt``
      are not on the atom); else the latest time in the deal; else 0 for every
      line, so the deal reads in compile order;
    * order: rank in the deal record's reading order (``_reading_rank``,
      ``c3_record.reading_order``), the rule the training export uses;
    * table_row: ``c3_record.table_geometry`` of the line's table ref.

    Returns the deal and each atom's key, in the compile's order.
    """
    from ml.c3.data import DealExample

    deal_id, rows, keys = _rows(atoms, artifact_sha256)
    return DealExample.from_dict({"deal_id": deal_id, "company": "purtera", "atoms": rows}), keys


def _rows(atoms: list[Any], artifact_sha256: dict[str, str] | None = None,
          documents: list[dict[str, Any]] | None = None
          ) -> tuple[str, list[dict[str, Any]], list[str]]:
    """(deal id, one row per atom in compile order, each atom's key), built by
    the deal record standard (``c3_record.build_rows``), the same rules the
    training export uses: every line takes its DOCUMENT's arrival.

    ``documents`` is the compile's document list (``deal_documents``); without
    it (or for a document it lacks) the documents are read off the atoms, with
    each document's earliest own line time standing in for its arrival."""
    from c3_record import build_rows

    sha = artifact_sha256 if artifact_sha256 is not None else _DEAL.get("sha")
    atoms = _with_also(atoms)
    docs = list(documents if documents is not None else (_DEAL.get("documents") or []))
    have = {str(d.get("doc_id")) for d in docs}
    lines, keys, seen = [], [], set()
    extra: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    messages: dict[str, tuple] = {}
    said_to: dict[str, Any] = {}
    for i, a in enumerate(atoms):
        key = _key(a, i)
        if key in seen:
            key = f"{key}#{i}"
        seen.add(key)
        # The base refuses a read in which two lines share an atom id
        # (data.DealExample.from_dict), and a compile can hold duplicates at
        # this point (ids are made unique later): 010115 read nothing at all
        # and typed every line with the LLM. A repeat carries its unique key.
        aid = str(getattr(a, "id", "") or "")
        if aid in seen_ids:
            aid = key
        seen_ids.add(aid)
        keys.append(key)
        d = _doc_id(a)
        if d not in have:
            e = extra.setdefault(d, {"doc_id": d, "kind": "", "sent_at": None, "authored_at": None})
            e["kind"] = e["kind"] or _event_kind(a)
            t = _own_time(a)
            if t is not None and (e["authored_at"] is None or t < e["authored_at"]):
                e["authored_at"] = t
                e["sent_at"] = t
        role, side = _speaker(a)
        msg = _message_of(a)
        if msg is not None:
            messages[aid] = msg
        to = _value(a).get("said_to")
        if isinstance(to, list) and to:
            said_to[aid] = to
        lines.append({"key": key, "text": str(getattr(a, "raw_text", "") or ""), "doc_id": d,
                      "section": _section(a), "order": float(i), "speaker_role": role,
                      "speaker_side": side, "atom_id": aid, **_sender_inputs(a), **_geometry(a)})
    rank = _reading_rank(atoms, docs)
    if rank is not None:
        for ln, r in zip(lines, rank):
            ln["order"] = float(r)
    for d, e in extra.items():
        if not e["kind"]:
            first = next(a for a in atoms if _doc_id(a) == d)
            e["kind"] = _file_kind(first, sha)
    rows = build_rows(lines, docs + list(extra.values()), latest_fallback=0.0, timeline=_DEAL.get("stage_timeline"))
    _STATUS["rows"] = {"documents": len(docs), "documents_from_atoms": len(extra),
                       "times": len({r["entered_at"] for r in rows})}
    if view_name():
        rows = _view_rows(rows, docs + list(extra.values()), messages, said_to)
    deal_id = next((str(getattr(a, "project_id", "") or "") for a in atoms
                    if getattr(a, "project_id", None)), "") or "live"
    return deal_id, rows, keys


def view_name() -> str:
    """This compile's read of the deal (``c3_record.view``): ``quoting`` when the run has an as_of,
    ``planning`` otherwise; '' when ``SOWSMITH_C3_VIEW`` is off (every line read, every head asked)."""
    return str(_DEAL.get("view") or "")


def views_enabled() -> bool:
    """``SOWSMITH_C3_VIEW`` (default on): the read is the training view the run is in."""
    return os.environ.get("SOWSMITH_C3_VIEW", "1").strip().lower() not in ("0", "false", "no", "off")


def set_other_job(doc_ids: Any) -> None:
    """Documents set aside as another job (``c3_record.document_job.set_aside``): never in either view's
    read, as training drops them (S5). The compile's own lines are unchanged."""
    _DEAL["other_job"] = {str(d): "document_job other_job" for d in (doc_ids or ()) if d}


def view_drops_other_job() -> bool:
    """``SOWSMITH_C3_VIEW_OTHER_JOB`` (default on, dev included): with document_job_scope off, the view's
    read still leaves out the documents the base's document head sets aside as another job, by the one
    rule training's views drop them by (``c3_record.document_job.set_aside``)."""
    return os.environ.get("SOWSMITH_C3_VIEW_OTHER_JOB", "1").strip().lower() not in ("0", "false", "no", "off")


def other_job_by_head(atoms: list[Any], *, deal_name: str = "", crm: dict[str, Any] | None = None,
                      floor: float | None = None) -> set[str]:
    """Documents the base's ``jdg:document_job`` reading sets aside (``c3_record.document_job.set_aside``:
    a sure other_job, never a drawing on the model's word, never a document whose own lines name the
    deal). Empty when the base did not read the deal."""
    from c3_record.document_job import deal_anchors, is_drawing, names_deal, set_aside

    got = documents(atoms)
    if not got:
        return set()
    if floor is None:
        from app.core.document_job_scope import c3_aside

        floor = c3_aside()
    bar = floor
    anchors = deal_anchors(deal_name, crm or {})
    texts: dict[str, list[str]] = {}
    names: dict[str, str] = {}
    for a in atoms:
        d = _doc_id(a)
        texts.setdefault(d, []).append(str(getattr(a, "raw_text", "") or ""))
        names.setdefault(d, _ref(a, "filename"))
    out = set()
    for d, heads in got.items():
        v = heads.get(DOCUMENT_JOB_KEY)
        if not v:
            continue
        if set_aside(v[0], taught=False, confidence=float(v[1] or 0.0), floor=bar,
                     drawing=is_drawing([names.get(d, "")]), named=names_deal(texts.get(d, ()), anchors)):
            out.add(d)
    return out


def _message_of(atom: Any) -> tuple | None:
    """The message a mail line was written in, as training's provenance reads it off the page's line
    (``message``: author, sentAt, quoted, index) or the envelope's locator: (author, at, quoted, index)."""
    from c3_record.provenance import parse_when, zoned

    v, loc = _value(atom), _locator(atom)
    et = v.get("email_thread") if isinstance(v.get("email_thread"), dict) else {}
    msg = et.get("message") if isinstance(et.get("message"), dict) else {}
    index = msg.get("index") if msg.get("index") is not None else loc.get("message_index")
    if index in (None, "") and "message_index" not in loc:
        return None
    author = msg.get("author") or loc.get("sender") or ""
    sent = msg.get("sentAt") or msg.get("sent_at") or loc.get("sent_at")
    quoted = msg.get("quoted") if msg.get("quoted") is not None else loc.get("quoted")
    return (str(author or ""), parse_when(sent), bool(quoted), index, zoned(sent))


def _view_rows(rows: list[dict[str, Any]], docs: list[dict[str, Any]], messages: dict[str, tuple],
               said_to: dict[str, Any]) -> list[dict[str, Any]]:
    """The rows of this compile's view (c3_record.provenance / c3_record.view, the rules training's
    views use): a quoting read holds the customer's and partner's lines before our first pricing (cut
    earlier by the run's as_of), never our own paper or a Deal Kit line; a planning read every line before
    our kickoff once the deal is won. Neither holds another job's documents. Each kept row gets the view's
    stage clamp and its ``view``. The compile's atoms are unchanged: a line outside the view has no reading."""
    from c3_record import crm_marks
    from c3_record import provenance as prov
    from c3_record import view as cview
    from c3_record.provenance import parse_when

    v = view_name()
    from c3_record.sender import epoch

    if not any((epoch(r.get("entered_at")) or 0.0) > 0 for r in rows):
        # A deal with no time anywhere (training refuses one): no cut can be placed, so the read is every
        # line, as before views.
        _STATUS["view"] = {"view": v, "undated": True, "inputs": len(rows), "out": 0}
        for r in rows:
            r["view"] = v
        return rows
    docinfo: dict[str, dict[str, Any]] = {}
    for d in docs:
        o = d.get("origin") if isinstance(d.get("origin"), dict) else {}
        direction = str(o.get("direction") or "")
        sender = str(o.get("sender_email") or o.get("sender_name") or "")
        kind = str(d.get("kind") or "").strip().lower()
        docinfo[str(d.get("doc_id") or "")] = {
            "direction": direction, "sender": sender, "role": str(d.get("role") or ""),
            "kind": "" if kind == "file" else kind,
            "carrier": ("email", direction, sender) if o.get("via") == "email" else None,
            "carrier_at": parse_when(o.get("sent_at")) if o.get("via") == "email" else 0.0,
            "message": kind in ("email", "note"), "filename": str(d.get("filename") or "")}
    lines = [{"label_key": r["key"], "atom_id": r.get("atom_id") or r["key"], "doc_id": r.get("doc_id"),
              "text": r.get("text"), "entered_at": r.get("entered_at"), "doc_kind": r.get("doc_kind"),
              "speaker_side": r.get("speaker_side"), "section": r.get("section")} for r in rows]
    texts_of = {x["atom_id"]: (str(x["text"] or ""), messages[x["atom_id"]][4]) for x in lines
                if x["atom_id"] in messages}
    other = dict(_DEAL.get("other_job") or {})
    P = prov.decide(lines, docinfo, {k: m[:4] for k, m in messages.items()}, texts_of=texts_of, to_of=said_to, other_job=other,
                    source="compile")
    dk = prov.pricing_workbook_keys(P, lines)
    marks = crm_marks(_DEAL.get("stage_timeline"))
    end = None
    if v == cview.QUOTING:
        keep = prov.quoting_inputs(P, lines, exclude=dk, as_of=_DEAL.get("as_of"))
    else:
        won = min((x["arrival"] for x in P.doc.values() if x["ours"] and x["kind"] in ("sow_signed", "signed_sow",
                                                                                     "purchase_order")), default=None)
        end = cview.planning_end(lines, cut=P.cut, timeline_marks=marks, won_at=won)
        keep = cview.planning_inputs(lines, end, exclude=dk, other_job=other)
    _DEAL["view_prov"] = P          # the last view's provenance (audits read it; never sent)
    kept = [rows[i] for i in keep]
    cview.stamp(kept, v, crm=marks is not None, cut=P.cut)
    _STATUS["view"] = {"view": v, "cut": None if P.cut == float("inf") else P.cut, "cut_why": P.cut_why,
                       "plan_end": None if end in (None, float("inf")) else end, "inputs": len(kept),
                       "out": len(rows) - len(kept), "ours": len(P.ours_atoms), "deal_kit": len(dk),
                       "other_job_docs": len(other)}
    return kept


def _geometry(atom: Any) -> dict[str, int]:
    """Where a table-row line sits (``c3_record.table_geometry``), from the
    table ref the labeling page also stamps (``_atom_table_ref``: the page's
    ``tableRef``, which training's ``table_row`` is read from)."""
    try:
        from app.core.typed_atom_classifier import _atom_table_ref
        from c3_record import table_geometry

        return table_geometry(_atom_table_ref(atom))
    except Exception:  # noqa: BLE001 - no geometry is a line read without it, never a failed read
        return {}


def also_read(atoms: list[Any]) -> None:
    """Lines the deal's read must hold that are not in the compile's list:
    the copies a dedup folded before the heads read (``head_fold``) and the
    lines an admission regex held aside. Each is read in its own document's
    context, so the heads decide about it; the compile's list is unchanged.
    Every read after this (``documents``, typing's ``classify``, the late
    re-read) appends them after the compile's own lines."""
    _DEAL["also"] = list(atoms or [])


def _with_also(atoms: list[Any]) -> list[Any]:
    extra = _DEAL.get("also") or []
    if not extra:
        return list(atoms)
    have = {id(a) for a in atoms}
    ids = {str(getattr(a, "id", "") or "") for a in atoms}
    return list(atoms) + [a for a in extra if id(a) not in have and str(getattr(a, "id", "") or "") not in ids]


def _reading_rank(atoms: list[Any], docs: list[dict[str, Any]]) -> list[int] | None:
    """Each atom's rank in the deal record's reading order
    (``c3_record.reading_order``), the rule the training export orders the
    labeling page's lines by: documents by authored time, ties by manifest
    place; inside one, the page's own order (``c3_record.page_position``:
    message, page, block, line, sentence, column), a workbook's sheets one
    after another. Ties fall to the envelope's order (``_in_reading_order``,
    which is the page's envelope order), else the compile's."""
    try:
        from c3_record import is_email_file, page_group, page_position, reading_order

        envelope = list(atoms)
        if docs and all("pos" in d for d in docs):
            try:
                from app.core.orbitbrief_envelope import _in_reading_order

                reading = [{"artifact_id": d["doc_id"], "authored_at": d.get("authored_at") or ""}
                           for d in sorted(docs, key=lambda d: d["pos"])]
                envelope = _in_reading_order(list(atoms), reading)
            except Exception:
                envelope = list(atoms)
        lines = [{"doc_id": _doc_id(a),
                  "within": page_position(_locator(a), _value(a), getattr(a, "raw_text", "") or "",
                                          email=is_email_file(_ref(a, "artifact_type"), _ref(a, "filename"))),
                  "group": page_group(_locator(a))} for a in envelope]
        rank = reading_order(lines, [{"doc_id": d.get("doc_id"), "authored_at": d.get("authored_at"),
                                      "pos": d.get("pos")} for d in docs or []])
        place = {id(a): r for a, r in zip(envelope, rank)}
        return [place[id(a)] for a in atoms]
    except Exception:
        return None


# ------------------------------------------------------------ the deal

#: This compile's documents (``deal_documents``) and file fingerprints, set
#: once by the compiler so every read of the deal sees the same record, and
#: the readings already made, so the deal is read once per compile.
_DEAL: dict[str, Any] = {}


def document_job_enabled() -> bool:
    """``SOWSMITH_C3_DOCUMENT_JOB`` is on or shadow (document_job_scope.c3_mode)."""
    return os.environ.get("SOWSMITH_C3_DOCUMENT_JOB", "").strip().lower() in ("1", "true", "yes", "on", "shadow")


#: The line readings brief v2 (``c3_brief``) uses, asked for in the compile's
#: one read of the deal when ``SOWSMITH_C3_BRIEF`` is on.
BRIEF_LINE_HEADS = ("read:co_action", "read:co_reason", "read:sow_section", "read:noise_class",
                    "read:scope_facet", "col:wants", "read:intake_gap", "read:needed_by",
                    "read:blocked_on", "read:superseded", "read:task_owner_role",
                    # Site role for site_geo_fallback (line_reading): on the labeled
                    # deals the base was right 241/243, the LLM/parser 234/243.
                    "jdg:site_role", "jdg:geo_mention",
                    # Who furnishes a priced line (c3_brief._customer_furnished).
                    "col:supplier")


def line_reading(atom_id: str, head: str) -> tuple[str, float] | None:
    """This compile's base reading of one line, or None when the line was not
    read (created after the read) or the head was not asked."""
    v = ((_DEAL.get("reads") or {}).get(str(atom_id)) or {}).get(head)
    return (str(v[0]), float(v[1] or 0.0)) if v and v[0] is not None else None


#: Type answers that are the type head's reject: in training, chatter and
#: "_keep" lines carry their drop in label_type (rejected=true) and never a
#: read:co_action label, so co_action is answering outside what it learned on
#: them (010388 "Yesterday's game was nuts.": small_talk 0.94, co_action keep
#: 0.99). Whose call a line's keep is: the type head's here, co_action's on the
#: lines the type head reads as substantive.
CHATTER_ANSWERS = frozenset({"small_talk", "_keep"})


def type_reading(atom_id: str) -> tuple[str, float] | None:
    """This compile's type head (col:label_type) reading of one line."""
    v = (_DEAL.get("types") or {}).get(str(atom_id))
    return (str(v[0]), float(v[1] or 0.0)) if v and v[0] is not None else None


def is_chatter_reading(reading: tuple[str, float] | None) -> bool:
    """The type head reads the line as small_talk or _keep at its SURE bar."""
    return bool(reading) and reading[0] in CHATTER_ANSWERS and float(reading[1] or 0.0) >= sure_for(reading[0])


def type_head_drops(atom_id: str) -> bool:
    """True when the type head drops this line (``is_chatter_reading``): no
    keep head reading rescues it."""
    return is_chatter_reading(type_reading(atom_id))


def stamp_type_reads(atoms: list[Any]) -> int:
    """Write the type head's reading on every line it read
    (``value.c3_read = {"type", "p"}``), acted on or not, so the SOW writer
    and audits see it without a re-read. Returns lines stamped."""
    n = 0
    for a in atoms:
        got = type_reading(str(getattr(a, "id", "") or ""))
        v = getattr(a, "value", None)
        if got is None or not isinstance(v, dict):
            continue
        v["c3_read"] = {"type": got[0], "p": round(got[1], 4)}
        n += 1
    return n


def brief_enabled() -> bool:
    return os.environ.get("SOWSMITH_C3_BRIEF", "").strip().lower() in _TRUE


def brief_questions() -> list[str]:
    """The catalog questions the read asks of the whole deal (c3_record)."""
    try:
        from c3_record.questions_catalog import texts

        return texts()
    except Exception:
        return []


#: Table verdicts the read asks (predict._judge_tables): whether a table is the deal's site roster or a
#: sheet, per table (doc_id||section). The Deal Kit reads a site_roster table as one site list.
TABLE_HEADS = ("jdg:roster", "jdg:sheet")


def brief_line_heads() -> list[str]:
    """``BRIEF_LINE_HEADS`` this compile's view asks (``c3_record.view.heads``): a quoting read asks no PM
    head, a planning read no quote or SOW head."""
    from c3_record.view import heads

    return heads(view_name(), BRIEF_LINE_HEADS)


def _brief_body(body: dict[str, Any]) -> dict[str, Any]:
    if view_name():
        body["view"] = view_name()
    if brief_enabled():
        body.update(line_heads=brief_line_heads(), full=True, table_heads=list(TABLE_HEADS))
        qs = brief_questions()
        if qs:
            body["questions"] = qs
    return body


def _keep_full(got: dict[str, Any]) -> None:
    """Hold the read's line readings, links, deal answers, question answers
    and table verdicts for the brief. A later read of the deal replaces the
    links, questions and tables of an earlier one: a later line can change an
    earlier reading, so the two never mix."""
    if "reads" in _DEAL:
        _DEAL["reads"].update(got.get("reads") or {})
        _DEAL["questions"] = list(got.get("questions") or [])
        if got.get("question_bar") is not None:
            _DEAL["question_bar"] = float(got["question_bar"])
        _DEAL["links"] = list(got.get("links") or [])
        _DEAL["deal"].update(got.get("deal") or {})
        _DEAL["tables"] = dict(got.get("tables") or {})


def full_read() -> dict[str, Any]:
    """This compile's readings beyond types: ``{"reads", "links", "deal"}``."""
    return {k: _DEAL.get(k) for k in ("reads", "links", "deal", "questions", "question_bar", "tables", "view")}


def set_deal(documents: list[dict[str, Any]] | None, artifact_sha256: dict[str, str] | None = None,
             stage_timeline: dict[str, Any] | None = None, as_of: Any = None) -> None:
    """``stage_timeline``: the deal's ``crm.stage_timeline`` from the parser manifest (the envelope's,
    cut to the run's as_of date), which ``c3_record.build_rows`` reads for ``stage_at``. ``as_of``: the
    run's cutoff (manifest ``context.as_of``): a run with one is a quoting read, one without a planning
    read (``view_name``; ``SOWSMITH_C3_VIEW=0`` reads every line as before)."""
    _DEAL.clear()
    _DEAL.update(documents=list(documents or []), sha=dict(artifact_sha256 or {}), types={}, docs={},
                 reads={}, links=[], deal={}, questions=[], also=[], tables={},
                 stage_timeline=stage_timeline if isinstance(stage_timeline, dict) else None)
    if views_enabled() and documents is not None:
        from c3_record.view import of_run

        _DEAL.update(view=of_run(as_of), as_of=as_of or None)


def _delivered_at(sha256: str) -> Any:
    rec = _lifecycle(sha256)
    for d in rec.get("delivered") or []:
        if isinstance(d, dict) and d.get("ts"):
            return d["ts"]
    return None


def deal_documents(atoms: list[Any], fingerprints: dict[str, dict[str, Any]],
                   project_dir: Any = None) -> list[dict[str, Any]]:
    """The compile's documents in the deal record's shape, in arrival order.

    ``fingerprints``: artifact id -> ``{"filename", "sha256"}``. Arrival comes
    from the manifest Purpulse writes: the email or note that carried a file
    (``carried_by.sent_at``, else that carrier's authored time), and an
    event's own authored time; a file nothing carried has none (it takes the
    arrival of the document before it in reading order, ``c3_record.
    document_times``, as the training export does). Kind is the labeling
    page's event kind (``c3_record.event_kind``), else the file's type by its
    bytes (``_file_kind``); role is the lifecycle's (the page's ``docRole``);
    origin is ``doc_origin.origin_for`` with the deal roster's names."""
    prov: dict[str, dict[str, Any]] = {}
    index: dict[str, dict[str, Any]] = {}
    if project_dir:
        try:
            from app.core.orbitbrief_envelope import _load_manifest_provenance

            prov = _load_manifest_provenance(project_dir) or {}
        except Exception:
            prov = {}
        try:
            from app.core.document_job_scope import manifest_index

            index = manifest_index(project_dir) or {}
        except Exception:
            index = {}
    from c3_record import event_kind

    first: dict[str, Any] = {}
    own: dict[str, list[Any]] = {}
    for a in atoms:
        d = _doc_id(a)
        first.setdefault(d, a)
        own.setdefault(d, []).append(a)
    # A document is an email, note or meeting by the labeling page's rule
    # (c3_record.event_kind: its type, its name, its own atoms), which is the
    # kind training reads.
    event = {d: event_kind(_ref(a, "artifact_type"), (fingerprints or {}).get(d, {}).get("filename")
                           or _ref(a, "filename"), [_value(x) for x in own[d]]) for d, a in first.items()}
    sha = {k: str(v.get("sha256") or "") for k, v in (fingerprints or {}).items() if v.get("sha256")}
    by_name = {str(fp.get("filename") or ""): d for d, fp in (fingerprints or {}).items()}
    prov = _as_envelope_documents(prov, own, by_name, sha)
    # The HubSpot note each file was attached to (note_attachments), which the envelope's own
    # documents carry as ``hubspot_note`` and the labeling page's origin reads ("Uploaded internally by
    # Trent Torrence · Jul 27 note"): without it every note-attached file read "Uploaded internally",
    # with no sender and no arrival (11,577 speaker and 2,988 arrival differences from training, run E).
    try:
        from app.core.note_attachments import note_attachment_links

        _links = note_attachment_links(project_dir) if project_dir else None
        for name, link in ((_links.attachments if _links else {}) or {}).items():
            if name in prov and isinstance(link, dict):
                prov[name].setdefault("hubspot_note", link)
    except Exception:  # noqa: BLE001 - no links is today's behaviour
        pass
    names = _roster_names(atoms, prov)
    docs = []
    for pos, (d, fp) in enumerate((fingerprints or {}).items()):
        name = str(fp.get("filename") or "")
        p = prov.get(name) or {}
        authored = (p.get("authored_at") or (index.get(name) or {}).get("authored_at")
                    or _own_date(own.get(d) or [], event.get(d) or "") or None)
        cb = p.get("carried_by") if isinstance(p.get("carried_by"), dict) else {}
        carrier = prov.get(str(cb.get("filename") or "")) or {}
        sent = (cb.get("sent_at") or carrier.get("authored_at")) if cb else None
        if not sent and sha.get(d):
            # doc_origin.origin_for's last rule: the message the lifecycle
            # matched as delivering the file (live 000002: the SOW draft
            # arrived with the note at 23:32:57.777, as the page had it).
            sent = _delivered_at(sha[d])
        kind = event.get(d) or (_file_kind(first[d], sha) if d in first else "file")
        # The lifecycle's role sentence, the page's docRole: whose paper it is
        # (c3_record.sender.is_ours reads it for the stage).
        role = str(_lifecycle(sha.get(d)).get("role") or "")
        origin = _origin(p, prov, first.get(d), names)
        # The arrival training reads is the page's ``origin.sent_at`` (export.py) and nothing else: the
        # carrier's time, a HubSpot note's when the file came on one, the delivering message's when the
        # lifecycle matched one (doc_origin.origin_for). A file nothing delivered has none and takes the
        # arrival of the document before it (c3_record.document_times), as in training -- not the
        # lifecycle's own delivered_at guess (010050's 2022 predictive survey, 207 lines a day apart).
        sent = origin.get("sent_at") if origin else sent
        docs.append({"doc_id": d, "kind": kind, "sent_at": sent, "authored_at": authored,
                     "filename": name, "pos": pos, "role": role, "origin": origin})
    from c3_record import EVENT_KINDS

    def arrival(doc):
        t = doc["sent_at"] or (doc["authored_at"] if doc["kind"] in EVENT_KINDS else None)
        return _when(t)

    # Arrival order: documents with a known arrival by time; the rest after
    # the document before them in the manifest, so they inherit its arrival.
    keyed, prev = [], None
    for doc in docs:
        t = arrival(doc)
        prev = t if t is not None else prev
        keyed.append(((prev if prev is not None else float("-inf")), doc["pos"], doc))
    keyed.sort(key=lambda x: (x[0], x[1]))
    return [doc for _, _, doc in keyed]


def _as_envelope_documents(prov: dict[str, dict[str, Any]], own: dict[str, list[Any]],
                           by_name: dict[str, str], sha: dict[str, str] | None = None
                           ) -> dict[str, dict[str, Any]]:
    """The manifest provenance with what the envelope adds to each document
    before it reads the document's origin (``orbitbrief_envelope``): from its
    own atoms, an email's thread block (its MIME sender), the sender a
    forwarded chain started with and a HubSpot note's author; its lifecycle
    record; then the envelope's own passes over the documents -- who sent the
    message that delivered a file (``_resolve_delivered_by``), a file's
    direction from its originator, and the note that captions an upload
    (``_link_caption_notes``). ``doc_origin.origin_for`` reads these on the
    page's side."""
    try:
        from app.core.orbitbrief_envelope import (_direction_from_originator, _document_thread,
                                                  _link_caption_notes, _note_author, _originating_sender,
                                                  _resolve_delivered_by)
    except Exception:
        return prov
    out: dict[str, dict[str, Any]] = {}
    for name, p in prov.items():
        q = dict(p)
        d = by_name.get(name, "")
        mine = own.get(d) or []
        art = _ref(mine[0], "artifact_type").strip().lower() if mine else ""
        q.update(artifact_id=d, filename=name, artifact_type=art)
        try:
            if mine:
                q.setdefault("email_thread", _document_thread(mine, artifact_id=d, is_message=art == "email"))
            if mine and (p.get("attachment_ids") or []):
                q.setdefault("originated_by", _originating_sender(mine, artifact_id=d))
            if mine:
                for k, v in _note_author(mine, d).items():
                    q.setdefault(k, v)
        except Exception:  # noqa: BLE001 - what cannot be read is left as the manifest has it
            pass
        lc = _lifecycle((sha or {}).get(d))
        if lc:
            q.setdefault("lifecycle", lc)
        out[name] = q
    docs = list(out.values())
    compact = {"atoms": [{"artifact_id": d, "structured": _value(a)} for d, xs in own.items() for a in xs]}
    for step in (lambda: _resolve_delivered_by(docs), lambda: _direction_from_originator(docs, None),
                 lambda: _link_caption_notes(docs, compact, None)):
        try:
            step()  # in place, in the envelope's order
        except Exception:  # noqa: BLE001
            pass
    return out


def _roster_names(atoms: list[Any], prov: dict[str, dict[str, Any]]) -> dict[str, str]:
    """The deal's people, address -> name, as the envelope's ``deal_roster``
    has them (the labeling page passes ``rosterNames(env)`` to the origin of
    every document): message senders and signatures."""
    try:
        from app.core.deal_roster import build_deal_roster

        roster = build_deal_roster(atoms=list(atoms), documents=[
            {"sender_email": p.get("sender_email"), "filename": name} for name, p in prov.items()])
    except Exception:
        return {}
    return {str(p["email"]).lower(): str(p["name"]) for p in roster.get("people") or []
            if isinstance(p, dict) and p.get("email") and p.get("name")}


def _own_date(own: list[Any], kind: str) -> Any:
    """The page's fallback for a document's authored time: an email's thread
    date, a note's own date (atom-labeling.js: ``th.date``, ``note.date``)."""
    from c3_record.reading import is_note_meta

    for a in own:
        v = _value(a)
        if kind == "email":
            et = v.get("email_thread")
            if isinstance(et, dict) and et.get("thread_id") and et.get("date"):
                return et["date"]
        elif kind == "note" and is_note_meta(v):
            return v.get("date") or None
    return None


def _origin(p: dict[str, Any], prov: dict[str, dict[str, Any]], atom: Any = None,
            names: dict[str, str] | None = None) -> dict[str, Any]:
    """Who brought a document in, as the labeling page's ``origin`` says it:
    ``doc_origin.origin_for`` on the manifest's provenance (an attachment's
    is the email or note that carried it; an email's is its own sender), with
    the deal roster's names (``names``: address -> name, as the page passes
    them), so a sender known only by address still has a name a quoted
    message author can be matched to."""
    try:
        from app.core.doc_origin import origin_for

        art = (_ref(atom, "artifact_type") if atom is not None else "") or ""
        doc = {"carried_by": p.get("carried_by"), "direction": p.get("direction"),
               "sender_email": p.get("sender_email"), "authored_at": p.get("authored_at"),
               "artifact_type": "email" if art.strip().lower() in _EMAIL_TYPES else art,
               **{k: p[k] for k in ("email_thread", "originated_by", "note_author", "lifecycle", "delivered_by",
                                    "delivered_by_source", "forwarded_by", "hubspot_note") if p.get(k)}}
        o = origin_for(doc, prov, names or {})
    except Exception:  # noqa: BLE001 - an origin that cannot be read is no sender, never a failed read
        return {}
    return {k: o.get(k) for k in ("direction", "sender_email", "sender_name", "sent_at", "via")}


def _url() -> str:
    return os.environ.get("SOWSMITH_C3_URL", "").strip().rstrip("/")


def _request(path: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    import urllib.request

    req = urllib.request.Request(_url() + path, method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    token = os.environ.get("SOWSMITH_C3_TOKEN", "").strip()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def quote_enabled() -> bool:
    """Deal Kit quote heads (POST /v1/quote on the C3 server). Off unless SOWSMITH_C3_QUOTE is on."""
    return os.environ.get("SOWSMITH_C3_QUOTE", "").strip().lower() in ("1", "on", "true", "yes")


def _text_type(atom: Any) -> tuple[str, str, str]:
    kind = getattr(atom, "atom_type", "")
    # A compile's atoms carry their words in raw_text (as _rows reads them); text is a fallback.
    text = getattr(atom, "raw_text", None) or getattr(atom, "text", "")
    return (str(getattr(atom, "id", "") or ""), str(text or ""), str(getattr(kind, "value", kind) or ""))


def quote(atoms: list[Any], brief: dict[str, Any], deal_id: str = "") -> dict[str, Any]:
    """The quote heads' labor proposal for this deal: task line, job type, techs, hours per tech,
    rate line and price, each with its reason. Sends the final atoms and the brief's sections and
    job details; returns {} on any failure so a compile never waits on it."""
    if not quote_enabled() or not _url():
        return {}
    deal = brief.get("deal") or {}
    rows = []
    for a in atoms:
        aid, text, kind = _text_type(a)
        if text.strip():
            rows.append({"id": aid, "text": text[:4000], "type": kind})
    # The deal's site list count (the same list the envelope's site_readiness shows), so the heads'
    # "N places: check the site list" flag can fire. It was a literal 0, so it never did.
    try:
        from app.core.declared_scope import _found_site_count

        site_count = int(_found_site_count(atoms) or 0)
    except Exception:  # noqa: BLE001 - a count that cannot be read is sent as 0, as before
        site_count = 0
    body = {"deal": {"deal_id": deal_id, "deal_name": deal.get("deal_name") or "", "account": deal.get("account_name") or "",
                     "atoms": rows, "site_count": site_count,
                     "c3_brief": {"sections": {k: [{"id": x.get("id")} for x in (v or []) if isinstance(x, dict)]
                                               for k, v in (brief.get("sections") or {}).items()},
                                  "job_details": brief.get("job_details") or []}}}
    try:
        return _request("/v1/quote", body, timeout=float(os.environ.get("SOWSMITH_C3_QUOTE_S", "60")))
    except Exception as exc:  # noqa: BLE001 - best effort, like every other C3 call
        log.warning("quote heads unavailable: %s", type(exc).__name__)
        return {}


def warm() -> None:
    """Wake the base service when a compile starts, so a service that scaled
    to zero loads while the documents are read. Background, best-effort."""
    if mode() == "off" or not _url():
        return

    def _ping():
        try:
            _request("/health", None, timeout=_env_float("SOWSMITH_C3_WARM_S", 180.0))
        except Exception as exc:
            log.info("C3 base service warm-up: %s", type(exc).__name__)

    threading.Thread(target=_ping, daemon=True).start()


def deal_payload(deal: Any) -> dict[str, Any]:
    """The rows ``DealExample.from_dict`` reads back: what the base sees of
    each line, and nothing else."""
    return {"deal_id": deal.deal_id, "company": deal.company, "atoms": [
        {"key": a.key, "text": a.text, "entered_at": a.entered_at, "doc_id": a.doc_id,
         "doc_kind": a.doc_kind, "section": a.section, "order": a.order,
         "speaker_role": a.speaker_role, "speaker_side": a.speaker_side, "atom_id": a.atom_id,
         # Sender and stage (c3_record.sender), as the training export writes them.
         "speaker_org": getattr(a, "speaker_org", ""), "doc_direction": getattr(a, "doc_direction", ""),
         "stage_at": getattr(a, "stage_at", ""),
         # The read the line is in (c3_record.view), as training's views write it.
         **({"view": a.view} if getattr(a, "view", "") else {}),
         # Geometry: training writes table_row only (c3_record.table_geometry).
         **_table_row(a)}
        for a in deal.atoms]}


def _table_row(atom: Any) -> dict[str, int]:
    """``{"table_row": n}`` as the training export writes it, read back off
    the base's geometry (``ml.c3.data._geometry``: slot 4), else {}."""
    geo = tuple(getattr(atom, "geometry", ()) or ())
    return {"table_row": int(geo[4])} if len(geo) > 4 and geo[4] else {}


#: The base's name for a type the parser calls by another. ``work_scope_item``
#: is the parser's ``scope_item`` (the labelers' word for it); without this
#: every sure work_scope_item answer had no parser type and went to the LLM.
BASE_ALIAS = {"work_scope_item": "scope_item"}


def _aliased(types: dict[str, Any] | None) -> dict[str, Any]:
    return {k: ([BASE_ALIAS.get(str(v[0]), v[0]), *list(v[1:])] if v else v) for k, v in (types or {}).items()}


def read_key(rows: list[dict[str, Any]], *extra: Any) -> str:
    """What one read of the deal saw: every row the base reads (its key, text, document, arrival, order,
    sender, stage and view) and what was asked. A reading is reused only for the same key: a later line
    can change an earlier line's reading, so a subset of the atoms is not the same read."""
    import hashlib

    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r.get(k) for k in ("key", "text", "doc_id", "entered_at", "order", "speaker_side",
                                               "speaker_org", "doc_direction", "stage_at", "view", "doc_kind",
                                               "section")], default=str).encode())
    h.update(json.dumps([str(x) for x in extra]).encode())
    return h.hexdigest()


def _remote(deal_id: str, rows: list[dict[str, Any]], keys: list[str], ids: list[int]
            ) -> dict[int, tuple[str, float]]:
    import urllib.error

    budget = _env_float("SOWSMITH_C3_BUDGET_S", 90.0)
    started = time.monotonic()
    shared = _DEAL.get("types") or {}
    if keys and shared and _DEAL.get("read_key") == read_key(rows):
        # This compile already read exactly this deal (document_job_scope, on
        # the same record): the readings stand. Any change -- a line added,
        # dropped or rewritten -- reads again.
        _STATUS.update(state="read", reason="shared", remote=True, lines=len(rows), read_ms=0.0)
        return {i: (str(shared[k][0]), float(shared[k][1] or 0.0)) for i, k in zip(ids, keys)
                if shared[k] and shared[k][0] is not None}
    # Ask /health first, with a short timeout: a service that can't be
    # reached (or is still loading) costs the compile seconds, not the whole
    # read budget. 2026-10-15: a URL with no route to the Mac timed out the
    # full 100 s and the shadow join held the stage for 61 s of it.
    try:
        health = _request("/health", None, timeout=_env_float("SOWSMITH_C3_CONNECT_S", 5.0))
    except Exception as exc:
        code = exc.code if isinstance(exc, urllib.error.HTTPError) else None
        _STATUS.update(state="unreachable", remote=True, lines=len(rows),
                       reason=(f"HTTP {code}" if code else f"{type(exc).__name__}: {exc}")[:300],
                       read_ms=round((time.monotonic() - started) * 1000, 1))
        log.warning("C3 base service unreachable (%s); every atom goes to the LLM",
                    _STATUS["reason"])
        return {}
    if not health.get("ready"):
        _STATUS.update(state="warming", remote=True, lines=len(rows),
                       reason=str(health.get("error") or "still loading")[:300],
                       read_ms=round((time.monotonic() - started) * 1000, 1))
        return {}
    try:
        payload = {"deal_id": deal_id, "company": "purtera", "atoms": rows}
        body = _brief_body({"deal": payload, "budget_s": budget})
        got = _request("/read", body, timeout=budget + 10.0)
        _keep_full(got)
    except Exception as exc:
        code = exc.code if isinstance(exc, urllib.error.HTTPError) else None
        # 503: the service is still starting. That compile goes to the LLM;
        # the next one, a minute or two later, reads warm.
        state = {504: "over_budget", 503: "warming"}.get(code, "read_failed")
        _STATUS.update(state=state, remote=True, lines=len(rows),
                       reason=(f"HTTP {code}" if code else f"{type(exc).__name__}: {exc}")[:300],
                       read_ms=round((time.monotonic() - started) * 1000, 1))
        log.warning("C3 base service read failed (%s); every atom goes to the LLM",
                    _STATUS["reason"])
        return {}
    _STATUS.update(state="read", reason="", remote=True, lines=len(rows),
                   read_ms=round((time.monotonic() - started) * 1000, 1),
                   service_read_ms=got.get("read_ms"))
    types = _aliased(got.get("types"))
    if "types" in _DEAL:
        _DEAL["types"] = dict(types)
        _DEAL["read_key"] = read_key(rows)
    out: dict[int, tuple[str, float]] = {}
    for i, key in zip(ids, keys):
        t = types.get(key)
        if t and t[0] is not None:
            out[i] = (str(t[0]), float(t[1] or 0.0))
    return out


def _local(deal: Any, keys: list[str], ids: list[int]) -> dict[int, tuple[str, float]]:
    from ml.c3.predict import read_deal

    loaded = get_model()
    if loaded is None:
        return {}
    model, schema, device = loaded
    started = time.monotonic()
    budget = _env_float("SOWSMITH_C3_BUDGET_S", 90.0)
    try:
        doc = read_deal(model, deal, schema, max_atoms=MAX_ATOMS, device=device,
                        lines_only=True, deadline=started + budget, heads=("col:label_type",))
    except TimeoutError as exc:
        _STATUS.update(state="over_budget", reason=str(exc)[:300], lines=len(ids),
                       read_ms=round((time.monotonic() - started) * 1000, 1))
        log.warning("C3 type base read stopped (%s); every atom goes to the LLM", exc)
        return {}
    except Exception as exc:
        _STATUS.update(state="read_failed", reason=f"{type(exc).__name__}: {exc}"[:300],
                       lines=len(ids), read_ms=round((time.monotonic() - started) * 1000, 1))
        raise
    _STATUS.update(state="read", reason="", lines=len(ids),
                   read_ms=round((time.monotonic() - started) * 1000, 1))
    lines = doc.get("lines") or {}
    out: dict[int, tuple[str, float]] = {}
    for i, key in zip(ids, keys):
        got = ((lines.get(key) or {}).get("reads") or {}).get("col:label_type")
        if got and got.get("answer") is not None:
            out[i] = (BASE_ALIAS.get(str(got["answer"]), str(got["answer"])), float(got.get("p") or 0.0))
    return out


def _prepare(atoms: list[Any], artifact_sha256: dict[str, str] | None = None):
    """Take what the base reads from the atoms now, and return the read to
    run later: what it sees is fixed before the LLM touches any atom."""
    ids = [id(a) for a in atoms]
    _STATUS.pop("typing", None)  # this compile's counts only
    if _url():
        deal_id, rows, keys = _rows(atoms, artifact_sha256)
        return lambda: _remote(deal_id, rows, keys, ids)
    deal, keys = _build(atoms, artifact_sha256)
    return lambda: _local(deal, keys, ids)


def classify(atoms: list[Any], artifact_sha256: dict[str, str] | None = None
             ) -> dict[int, tuple[str, float]]:
    """``{id(atom): (label_type answer, probability)}`` for every atom read.

    Reads the whole deal once. Empty when the model is not loaded."""
    return _prepare(atoms, artifact_sha256)()


class Job:
    """A read running beside the compile (shadow): ``result`` waits for it,
    and any failure reads as nothing."""

    def __init__(self, run):
        self._out: dict[int, tuple[str, float]] = {}
        self._done = threading.Event()

        def _go():
            try:
                self._out = run()
            except Exception as exc:
                log.warning("C3 shadow read failed (%s)", type(exc).__name__)
            finally:
                self._done.set()

        threading.Thread(target=_go, daemon=True, name="c3-shadow-read").start()

    def result(self, timeout: float | None = None) -> dict[int, tuple[str, float]]:
        self._done.wait(timeout)
        return self._out if self._done.is_set() else {}


def start(atoms: list[Any], artifact_sha256: dict[str, str] | None = None) -> Job:
    """Begin the read in the background (shadow), so it overlaps the LLM
    instead of adding to the stage's time."""
    return Job(_prepare(atoms, artifact_sha256))


# ----------------------------------------------------------------- shadow

_LAST_SHADOW: dict[str, Any] = {}


def get_last_shadow() -> dict[str, Any]:
    return dict(_LAST_SHADOW)


def report_shadow(would: dict[int, tuple[Any, str, str, float, str]]) -> dict[str, Any]:
    """What the layer would have done beside what the compile did.

    ``would``: id(atom) -> (atom, type before, answer, p, "assign" | "fill" |
    "keep"). An assignment or a fill (the base's type, the LLM's values)
    agrees when the atom ended as that type; a keep when it ended as the type
    it had. No text leaves the compile."""
    def _t(a: Any) -> str:
        t = getattr(a, "atom_type", None)
        return str(getattr(t, "value", t) or "")

    agree = disagree = 0
    by_answer: dict[str, dict[str, int]] = {}
    # What the compile ended with where the layer would have said otherwise,
    # as "layer answer->compile type": the dev review reads these pairs.
    pairs: dict[str, int] = {}
    for a, before, answer, _p, act in would.values():
        want = answer if act in ("assign", "fill") else before
        hit = _t(a) == want
        agree += hit
        disagree += not hit
        slot = by_answer.setdefault(answer, {"agree": 0, "disagree": 0})
        slot["agree" if hit else "disagree"] += 1
        if not hit:
            pair = f"{answer}->{_t(a)}"
            pairs[pair] = pairs.get(pair, 0) + 1
    payload = {
        "event": "c3_type_shadow", "stage": "typed_atom_classification", "sure": sure(),
        "would_assign": sum(1 for v in would.values() if v[4] == "assign"),
        "would_fill": sum(1 for v in would.values() if v[4] == "fill"),
        "would_keep": sum(1 for v in would.values() if v[4] == "keep"),
        "agree": agree, "disagree": disagree, "by_answer": by_answer,
        "disagree_pairs": pairs,
    }
    _LAST_SHADOW.clear()
    _LAST_SHADOW.update(payload)
    try:
        print(json.dumps(payload, ensure_ascii=True), file=sys.stderr)
    except Exception:
        pass
    return payload


# ---------------------------------------------------------- final labels

#: Answers the base kept a line for that are parser types: once every stage has
#: run, the line takes the base's type (000002, 2026-10-15: 53 of the gold
#: small_talk / deal_state lines ended as scope_item though the base read them
#: right, because a keep left the type as it was).
FINAL_TYPES = frozenset({"small_talk", "deal_state"})
_STILL_UNTYPED = frozenset({"scope_item", "entity", "customer_instruction"})


def apply_final_labels(atoms: list[Any]) -> int:
    """After every stage: a line the base kept with a FINAL_TYPES answer, and
    that no stage has typed since, takes that type. Its type before stays an
    alternative for the labeler, as deal_chatter does. Returns lines typed."""
    from app.core.schemas import AtomType

    n = 0
    for a in atoms:
        v = getattr(a, "value", None)
        lab = v.get("c3_label") if isinstance(v, dict) else None
        if not isinstance(lab, dict) or lab.get("type") not in FINAL_TYPES:
            continue
        t = getattr(a, "atom_type", None)
        before = str(getattr(t, "value", t) or "")
        if before not in _STILL_UNTYPED:
            continue
        try:
            a.atom_type = AtomType(lab["type"])
        except ValueError:
            continue
        nv = dict(v)
        alt = list(nv.get("alt_atom_types") or [])
        if before not in alt:
            alt.append(before)
        nv["alt_atom_types"] = alt
        if lab["type"] == "small_talk":
            nv.setdefault("rejected_by", "c3")
        a.value = nv
        flags = list(getattr(a, "review_flags", None) or [])
        if "c3_label" not in flags:
            try:
                a.review_flags = flags + ["c3_label"]
            except Exception:
                pass
        n += 1
    m = _override_simple_types(atoms)
    _STATUS["final_labels"] = n
    _STATUS["final_overrides"] = m
    return n + m


#: Types a line can hold without values the parser extracted, so the base's
#: sure reading can replace one with another at the end of the compile.
_SIMPLE = frozenset({
    "requirement", "exclusion", "contract_term", "deal_metadata", "acceptance_criterion",
    "task", "change_order_rule", "constraint", "dependency", "mitigation", "compliance_rule",
    "submission_req", "addendum_qa",
})
_OVERRIDE_FROM = _SIMPLE | _STILL_UNTYPED | {"action_item", "open_question"}
_OVERRIDE_TO = _SIMPLE | {"small_talk", "deal_state"}


def _override_simple_types(atoms: list[Any]) -> int:
    """A line whose type the parser set by rule, before typing ever asked the
    base, takes the base's type where the base is sure (its one bar,
    ``sure_for``) and both types are simple. Validation deal 010043 (2026-10-15):
    on the gold-labeled lines this fixed 101 and broke none; at 0.85 the base
    reads 88% right where the parser is 45% on the same lines."""
    from app.core.schemas import AtomType

    readings = _DEAL.get("types") or {}
    if not readings:
        return 0
    n = 0
    for a in atoms:
        got = readings.get(str(getattr(a, "id", "") or ""))
        if not got or got[0] not in _OVERRIDE_TO:
            continue
        ans, p = str(got[0]), float(got[1] or 0.0)
        if p < sure_for(ans):
            continue
        t = getattr(a, "atom_type", None)
        before = str(getattr(t, "value", t) or "")
        if before == ans or before not in _OVERRIDE_FROM:
            continue
        try:
            a.atom_type = AtomType(ans)
        except ValueError:
            continue
        v = dict(getattr(a, "value", None) or {})
        alt = list(v.get("alt_atom_types") or [])
        if before not in alt:
            alt.append(before)
        v["alt_atom_types"] = alt
        v["c3_label"] = {"type": ans, "p": round(p, 3), "act": "override", "was": before}
        a.value = v
        flags = list(getattr(a, "review_flags", None) or [])
        if "c3_label" not in flags:
            try:
                a.review_flags = flags + ["c3_label"]
            except Exception:
                pass
        n += 1
    return n


# ---------------------------------------------------------- type authority
#
# The base decides every line's type when it is sure; the parser only splits
# lines and extracts values. Applied twice: before typing (every line, not only
# the ones routed to the LLM) and at the end of the compile, so a rule that
# re-typed a line in between does not stand against a sure reading. The rule's
# type is kept as an alternative for the labeler.


def authority_enabled() -> bool:
    return os.environ.get("SOWSMITH_C3_AUTHORITY", "").strip().lower() in _TRUE


def _authority_bar(answer: str) -> float:
    """The one bar every line's type answer must reach to act: the same
    ``sure_for`` typing uses (``SOWSMITH_C3_SURE``, per type by
    ``SOWSMITH_C3_SURE_BY_TYPE``). It was a second bar (0.9,
    ``SOWSMITH_C3_AUTHORITY_SURE``) for every line typing did not route, so
    a scope_item/entity line took the head's type at 0.5 and every other line
    needed 0.9: 1549 of 2183 read lines on the 8 audited deals were off the
    0.5 path (overrides.md point 7). The old variables are no longer read."""
    return sure_for(answer)


#: Never assigned by authority: a placeholder, or a type that only exists as
#: the parser's container for something not yet read.
_AUTHORITY_SKIP = frozenset({"_keep", "raw_utterance", "raw_table_row", "dropped_sheet", "entity"})
#: Types a line can take without extracted values.
_VALUELESS = _SIMPLE | frozenset({"small_talk", "deal_state", "list_header", "answered_question",
                                    "internal_question", "deal_summary", "source_caveat", "legend_entry",
                                    "certification",
                                    "commitment", "open_question", "action_item", "scope_item"})


def is_locked(atom: Any) -> bool:
    """True when the base decided this line's type (authority)."""
    v = getattr(atom, "value", None)
    lab = v.get("c3_label") if isinstance(v, dict) else None
    return isinstance(lab, dict) and lab.get("act") == "authority"


def base_decided(atom: Any) -> bool:
    """True when the base typed this line: authority, or typing's assign /
    keep / fill on the base's answer."""
    v = getattr(atom, "value", None)
    lab = v.get("c3_label") if isinstance(v, dict) else None
    return isinstance(lab, dict) and lab.get("act") in ("authority", "assign", "keep", "fill")


def reread_if_new(atoms: list[Any]) -> int:
    """Read the deal again when its lines changed after the base's read
    (signature splits, site backfills, geo mentions, a line dropped or
    rewritten): a line is read in its document's context, and a later line
    can change an earlier one's reading, so any change reads the whole deal
    again and its answers REPLACE the first read's. A re-read that fails
    leaves no readings rather than the stale ones. Returns the lines whose
    reading changed hands (new, rewritten or gone), 0 when nothing changed."""
    if not authority_enabled() or "types" not in _DEAL:
        return 0
    if not _DEAL.get("read_key"):
        return 0
    _deal_id, rows, _keys = _rows(atoms, _DEAL.get("sha"))
    if read_key(rows) == _DEAL.get("read_key"):
        return 0
    seen = set(_DEAL.get("types") or {})
    now = {r["key"] for r in rows}
    changed = len(now ^ seen) or len(rows)
    _DEAL.update(docs={}, docs_key=None, reads={}, links=[], deal={}, questions=[], question_bar=None, tables={},
                 types={}, read_key=None)
    documents(atoms)
    return changed


def settle_base_answers(atoms: list[Any]) -> tuple[int, int]:
    """Apply the base's answer to every line still carrying one unapplied
    (``value.c3_base_says``): a type with no fields is assigned, a type whose
    fields something reads is filled by the LLM, as typing does. On 010038
    56 lines kept a generic type although the base had them right (the OLD and
    NEW site lines at 1.00, a CDW contact at 0.94). Returns (assigned, filled)."""
    from app.core.schemas import AtomType

    try:
        from app.core.typed_atom_classifier import _TAXONOMY, _c3_fill_types, _fill_values
    except Exception:
        return 0, 0
    sha = _DEAL.get("sha")
    fills, assigned = [], 0
    for a in atoms:
        v = getattr(a, "value", None)
        if not isinstance(v, dict) or isinstance(v.get("c3_label"), dict):
            continue
        says = v.get("c3_base_says")
        if not isinstance(says, dict) or not says.get("type"):
            continue
        try:
            if skipped(a, sha):
                continue
        except Exception:
            pass
        ans, p = str(says["type"]), float(says.get("p") or 0.0)
        if p < _authority_bar(ans) or ans in _AUTHORITY_SKIP:
            continue
        try:
            t = AtomType(ans)
        except ValueError:
            continue
        if ans in _TAXONOMY and ans in _c3_fill_types():
            fills.append((a, (t, ans)))
            continue
        before = str(getattr(getattr(a, "atom_type", None), "value", getattr(a, "atom_type", "")) or "")
        a.atom_type = t
        a.value = {**v, "c3_label": {"type": ans, "p": round(p, 3), "act": "authority", "was": before}}
        assigned += 1
    filled = 0
    if fills:
        filled, _back = _fill_values(fills)
    return assigned, filled


_CITY = re.compile(r"\b([A-Z][a-z.']+(?:\s+[A-Z][a-z.']+){0,2}),\s*([A-Z]{2})\b")
_ADDRESS = re.compile(r"\b(\d{1,6})\s+[A-Za-z0-9][^\n|]*?\b[A-Z]{2}\s+\d{5}(?:-\d{4})?")


def own_site_keys(atoms: list[Any]) -> int:
    """A line the base types physical_site names its own place: it gets the
    site key of the address it states, and loses site keys of other street
    numbers. Keys are minted at parse time, so a line that became a site later
    had none of its own; on 010038 "NEW Site: 323 Lakeside Ave, Suite 403,
    Cleveland, OH 44113" carried only the OLD office's key, and the new office
    never reached the roster. Returns lines changed."""
    n = 0
    for a in atoms:
        t = getattr(a, "atom_type", None)
        if str(getattr(t, "value", t) or "") != "physical_site":
            continue
        text = str(getattr(a, "raw_text", "") or "")
        m = _ADDRESS.search(text)
        if not m:
            # A site the base reads that gives only its town ("Location is
            # Fairfield, CA.", 010440): the roster takes a site line only
            # with a site id and key, and one it had neither, so the deal's
            # only job site never reached it.
            c = _CITY.search(text)
            v = getattr(a, "value", None)
            if c and isinstance(v, dict) and not (v.get("id") or v.get("site_id")):
                slug = re.sub(r"[^a-z0-9]+", "_", f"{c.group(1)} {c.group(2)}".lower()).strip("_")
                keys = list(getattr(a, "entity_keys", None) or [])
                if not any(str(k).startswith("site:") for k in keys):
                    keys.append("site:" + slug)
                try:
                    name = str(v.get("name") or "")
                    if not name or not re.search(r",\s*[A-Z]{2}\b", name):
                        name = f"{c.group(1)}, {c.group(2)}"
                    a.value = {**v, "site_id": slug, "name": name}
                    a.entity_keys = keys
                    n += 1
                except Exception:
                    pass
            continue
        house = m.group(1)
        mine = "site:" + re.sub(r"[^a-z0-9]+", "_", m.group(0).lower()).strip("_")
        keys = list(getattr(a, "entity_keys", None) or [])
        kept = [k for k in keys if not k.startswith("site:")
                or not re.match(r"site:(\d{1,6})_", k) or re.match(rf"site:{house}_", k)]
        if mine not in kept:
            kept.append(mine)
        if kept != keys:
            try:
                a.entity_keys = kept
                n += 1
            except Exception:
                pass
    return n


def apply_authority(atoms: list[Any], *, final: bool = False) -> int:
    """Give each line the base's type where it is sure. Before typing
    (``final=False``) and again at the end (``final=True``: a locked line a
    rule re-typed goes back). Returns lines (re)typed."""
    from app.core.schemas import AtomType

    if not authority_enabled():
        return 0
    n = 0
    if final:
        # A line typing gave the base's type (filled or assigned) is the
        # base's decision too: a rule that re-typed it after typing loses. On
        # 010038 "NEW Site: 323 Lakeside Ave" was filled as physical_site
        # (base 0.999) and a later rule turned it back into deal_metadata, so
        # the new office never reached the roster.
        from app.core.schemas import AtomType as _AT

        for a in atoms:
            v = getattr(a, "value", None)
            lab = v.get("c3_label") if isinstance(v, dict) else None
            # "keep" too: the base agreed with the parser's type, so it is the
            # base's decision as much as a fill is ("Some additional
            # information from Raf-" was kept list_header at 0.945 and a later
            # rule made it scope_item, so it reached the brief's scope).
            if not isinstance(lab, dict) or lab.get("act") not in ("fill", "assign", "keep") or not lab.get("type"):
                continue
            t = getattr(a, "atom_type", None)
            if str(getattr(t, "value", t) or "") == lab["type"]:
                continue
            try:
                a.atom_type = _AT(lab["type"])
                n += 1
            except ValueError:
                continue
    readings = _DEAL.get("types") or {}
    if not readings:
        return n
    sha = _DEAL.get("sha")
    for a in atoms:
        got = readings.get(str(getattr(a, "id", "") or ""))
        if not got or got[0] in _AUTHORITY_SKIP:
            continue
        # A Deal Kit or estimating sheet is the owner's training data, not a
        # document the base reads (it never trained on one): its lines keep the
        # parser's type. Counting them was what made the base look worse than
        # the parser on 010043 (rate_card on 212 kit lines).
        try:
            if skipped(a, sha):
                continue
        except Exception:
            pass
        ans, p = str(got[0]), float(got[1] or 0.0)
        if p < _authority_bar(ans):
            continue
        t = getattr(a, "atom_type", None)
        before = str(getattr(t, "value", t) or "")
        v = dict(getattr(a, "value", None) or {}) if isinstance(getattr(a, "value", None), dict) else {}
        # At the end the base's answer wins for every line it read at its bar,
        # locked or not, valued or not: a later stage rebuilds a line's value
        # and loses its lock (010043's "Do you have the model of the camera?"
        # was read answered_question at 0.9985 and ended open_question, so
        # the brief made it a customer responsibility; 010388's "Nick
        # Robertow", stakeholder 0.99, ended deal_metadata). The rule's type
        # rides on the line as a hint (value.rule_type). A value type whose
        # values the line lacks is parked for the fill path (settle).
        if before == ans:
            if not final and not is_locked(a):
                v["c3_label"] = {"type": ans, "p": round(p, 3), "act": "authority", "was": before}
                a.value = v
            continue
        # A value-carrying type only where the parser's own value is that kind
        # (a rule demoted a row it had read as one); otherwise the base's answer
        # is recorded and typing's fill path gives it values.
        if ans not in _VALUELESS and str(v.get("kind") or "") != ans:
            v.setdefault("c3_base_says", {"type": ans, "p": round(p, 3)})
            a.value = v
            continue
        try:
            a.atom_type = AtomType(ans)
        except ValueError:
            # A base type the parser has no type for (answered_question,
            # list_header, rate_card, commitment ...): the line keeps its type,
            # and the base's answer rides on it for every reader after (the
            # brief leaves an answered question or a heading out).
            v["c3_base_type"] = {"type": ans, "p": round(p, 3)}
            a.value = v
            continue
        alt = list(v.get("alt_atom_types") or [])
        if before and before not in alt:
            alt.append(before)
        v["alt_atom_types"] = alt
        if final and before:
            v["rule_type"] = before
        v["c3_label"] = {"type": ans, "p": round(p, 3), "act": "authority", "was": before}
        if ans == "small_talk":
            v.setdefault("rejected_by", "c3")
        a.value = v
        flags = list(getattr(a, "review_flags", None) or [])
        if "c3_label" not in flags:
            try:
                a.review_flags = flags + ["c3_label"]
            except Exception:
                pass
        n += 1
    _STATUS["authority_final" if final else "authority"] = n
    return n


# ------------------------------------------------------- document judgments

#: The base's answer to "is this document about this deal's job"
#: (``document_job_scope``), asked of each whole document.
DOCUMENT_JOB_KEY = "jdg:document_job"
_DOC_STATUS: dict[str, Any] = {"state": "off"}


def document_status() -> dict[str, Any]:
    return dict(_DOC_STATUS)


def documents(atoms: list[Any], keys: tuple[str, ...] = (DOCUMENT_JOB_KEY,),
              artifact_sha256: dict[str, str] | None = None) -> dict[str, dict[str, tuple[str, float]]]:
    """``{doc_id: {head: (answer, p)}}`` for every document of the deal, from
    one read of the base (the service when ``SOWSMITH_C3_URL`` is set, else
    the local model). ``doc_id`` is ``_doc_id(atom)``. Empty on any failure:
    the caller then asks its own judge, exactly as without the base."""
    if not atoms:
        return {}
    import urllib.error

    started = time.monotonic()
    budget = _env_float("SOWSMITH_C3_BUDGET_S", 90.0)
    deal_id, rows, _keys = _rows(atoms, artifact_sha256)
    cache = read_key(rows, *keys)
    if _DEAL.get("docs") and _DEAL.get("docs_key") == cache:
        return dict(_DEAL["docs"])
    try:
        if _url():
            try:
                health = _request("/health", None, timeout=_env_float("SOWSMITH_C3_CONNECT_S", 5.0))
            except Exception as exc:
                _DOC_STATUS.update(state="unreachable", reason=f"{type(exc).__name__}"[:300],
                                   read_ms=round((time.monotonic() - started) * 1000, 1))
                return {}
            if not health.get("ready"):
                _DOC_STATUS.update(state="warming", reason="", read_ms=0.0)
                return {}
            payload = {"deal_id": deal_id, "company": "purtera", "atoms": rows}
            body = _brief_body({"deal": payload, "budget_s": budget, "document_heads": list(keys)})
            got = _request("/read", body, timeout=budget + 10.0)
            raw = got.get("documents") or {}
            if "types" in _DEAL:
                _DEAL["types"] = dict(_aliased(got.get("types")))
                _DEAL["read_key"] = read_key(rows)
            _keep_full(got)
        else:
            from ml.c3.predict import read_deal

            loaded = get_model()
            if loaded is None:
                _DOC_STATUS.update(state="no_model", reason="", read_ms=0.0)
                return {}
            model, schema, device = loaded
            deal, _ = _build(atoms, artifact_sha256)
            doc = read_deal(model, deal, schema, max_atoms=MAX_ATOMS, device=device, lines_only=True,
                            deadline=started + budget, heads=("col:label_type",), document_heads=keys)
            raw = {d: {k: [v["answer"], v.get("p")] for k, v in h.items()}
                   for d, h in (doc.get("documents") or {}).items()}
    except Exception as exc:
        code = exc.code if isinstance(exc, urllib.error.HTTPError) else None
        _DOC_STATUS.update(state={504: "over_budget", 503: "warming"}.get(code, "read_failed"),
                           reason=(f"HTTP {code}" if code else type(exc).__name__)[:300],
                           read_ms=round((time.monotonic() - started) * 1000, 1))
        log.warning("C3 document read failed (%s); the judge decides as before", _DOC_STATUS["reason"])
        return {}
    out: dict[str, dict[str, tuple[str, float]]] = {}
    for d, heads in raw.items():
        for k, v in (heads or {}).items():
            if v and v[0] is not None:
                out.setdefault(str(d), {})[k] = (str(v[0]), float(v[1] or 0.0))
    if "docs" in _DEAL:
        _DEAL["docs"] = dict(out)
        _DEAL["docs_key"] = cache
    _DOC_STATUS.update(state="read", reason="", documents=len(out),
                       read_ms=round((time.monotonic() - started) * 1000, 1))
    return out
