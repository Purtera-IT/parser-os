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
    SOWSMITH_C3_FALLBACK     llm (default) | never. ``llm`` is everything
                             above: an unreachable, warming, over-budget or
                             failing heads service reads as no answer and
                             every atom goes to the LLM. ``never``: no answer
                             ever depends on the service's load. Every heads
                             call (type read, document read, re-read, quote)
                             waits for the service -- retrying with backoff
                             (SOWSMITH_C3_RETRY_S 5 doubling to
                             SOWSMITH_C3_RETRY_MAX_S 60) and reading with a
                             generous timeout (SOWSMITH_C3_NEVER_READ_S,
                             at least 600 and the budget) -- and past
                             SOWSMITH_C3_WAIT_MAX_S (1800) or on a request the
                             service refuses (4xx but 408/429) raises
                             ``C3HeadsUnavailable``, which fails the compile.
                             The local model likewise raises on a failed load
                             or a deadline overrun. Content handoffs (a line
                             the heads abstain on, a type not taken, unsure
                             below the bar) still go to the LLM. Shadow reads
                             keep the ``llm`` behaviour: they decide nothing.

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
_CALC_NAME = re.compile(r"estimat\w*calc")
#: A file named as a site list ("Site List Project Bastion.xlsx") is one: training saw these as site_list.
_SITE_LIST_NAME = re.compile(r"site(list|roster)|locationlist|storelist")

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


#: Every document kind the base trained on: the lowercased lifecycle types
#: (document_lifecycle/taxonomy.py) the training files carry, plus the three
#: event kinds and ``file``, training's own bucket for a document of no known
#: type. Anything else reads as ``file``, so the model never sees a tag it
#: never trained on.
TRAINED_KINDS = frozenset({
    "email", "meeting", "note", "file", "sow_draft", "deal_kit", "site_survey",
    "estimating_calc", "sow_signed", "site_list", "runbook", "cost_breakdown", "w9",
    "purchase_order", "rfq", "rate_card", "engineering_design", "install_instructions",
    "credit_application", "sow", "intake_request", "checklist", "quote", "bom",
    "vendor_datasheet", "floor_plan", "case_manifest", "sow_template", "nda", "crm_note",
})
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


def _file_kind(atom: Any, artifact_sha256: dict[str, str] | None) -> str:
    """A file's lifecycle type, looked up by its bytes' sha256 as the envelope
    does; else ``deal_kit`` / ``estimating_calc`` by the file's name; else
    ``file``."""
    sha = (artifact_sha256 or {}).get(_doc_id(atom))
    if sha:
        try:
            from app.core.document_lifecycle.dataset import lookup

            rec = lookup(sha)
        except Exception:
            rec = None
        kind = str((rec or {}).get("type") or "").strip().lower()
        if kind:
            return kind if kind in TRAINED_KINDS else "file"
    name = re.sub(r"[^a-z]", "", _ref(atom, "filename").lower())
    if "dealkit" in name:
        return "deal_kit"
    if _CALC_NAME.search(name):
        return "estimating_calc"
    if _SITE_LIST_NAME.search(name):
        return "site_list"
    return "file"


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


#: The speaker values training saw; anything else is '' (as on ~95% of lines).
_ROLES = frozenset({"internal", "reseller", "unknown"})
_SIDES = frozenset({"ours", "theirs"})


def _speaker(atom: Any) -> tuple[str, str]:
    """(role, side): the page's ``saidBy``, which is ``value.said_by``. A
    transcript line's locator ``speaker_role`` is not it: training has '' on
    99.98% of meeting lines, and ``unknown`` there meant an email sender."""
    from c3_record import speaker

    by = _value(atom).get("said_by")
    if not isinstance(by, dict):
        return "", ""
    return speaker(by.get("role") or by.get("role_guess"), by.get("side"))


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
    * speaker_role / speaker_side: ``value.said_by`` role and side, else the
      transcript locator's ``speaker_role``;
    * entered_at: the locator's ``sent_at``; else the time of the document
      before it in arrival order (a document's own origin and ``authoredAt``
      are not on the atom); else the latest time in the deal; else 0 for every
      line, so the deal reads in compile order;
    * order: position in the compile.

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
    docs = list(documents if documents is not None else (_DEAL.get("documents") or []))
    have = {str(d.get("doc_id")) for d in docs}
    lines, keys, seen = [], [], set()
    extra: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
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
        lines.append({"key": key, "text": str(getattr(a, "raw_text", "") or ""), "doc_id": d,
                      "section": _section(a), "order": float(i), "speaker_role": role,
                      "speaker_side": side, "atom_id": aid})
    rank = _reading_rank(atoms, docs)
    if rank is not None:
        for ln, r in zip(lines, rank):
            ln["order"] = float(r)
    for d, e in extra.items():
        if not e["kind"]:
            first = next(a for a in atoms if _doc_id(a) == d)
            e["kind"] = _file_kind(first, sha)
    rows = build_rows(lines, docs + list(extra.values()), latest_fallback=0.0)
    _STATUS["rows"] = {"documents": len(docs), "documents_from_atoms": len(extra),
                       "times": len({r["entered_at"] for r in rows})}
    deal_id = next((str(getattr(a, "project_id", "") or "") for a in atoms
                    if getattr(a, "project_id", None)), "") or "live"
    return deal_id, rows, keys


def _reading_rank(atoms: list[Any], docs: list[dict[str, Any]]) -> list[int] | None:
    """Each atom's place in the envelope's reading order (``_in_reading_order``,
    the order the labeling page lists lines in, which is training's ``order``):
    documents by authored time, ties by their place in the compile manifest,
    then page, paragraph or row, line and sentence. None without the compile's
    documents; the atoms' own order stands."""
    if not docs or not all("pos" in d for d in docs):
        return None
    try:
        from app.core.orbitbrief_envelope import _in_reading_order

        reading = [{"artifact_id": d["doc_id"], "authored_at": d.get("authored_at") or ""}
                   for d in sorted(docs, key=lambda d: d["pos"])]
        place = {id(a): i for i, a in enumerate(_in_reading_order(list(atoms), reading))}
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


def brief_enabled() -> bool:
    return os.environ.get("SOWSMITH_C3_BRIEF", "").strip().lower() in _TRUE


def brief_questions() -> list[str]:
    """The catalog questions the read asks of the whole deal (c3_record)."""
    try:
        from c3_record.questions_catalog import texts

        return texts()
    except Exception:
        return []


def _brief_body(body: dict[str, Any]) -> dict[str, Any]:
    if brief_enabled():
        body.update(line_heads=list(BRIEF_LINE_HEADS), full=True)
        qs = brief_questions()
        if qs:
            body["questions"] = qs
    return body


def _keep_full(got: dict[str, Any]) -> None:
    """Hold the read's line readings, links, deal answers and question
    answers for the brief."""
    if "reads" in _DEAL:
        _DEAL["reads"].update(got.get("reads") or {})
        _DEAL["questions"].extend(got.get("questions") or [])
        if got.get("question_bar") is not None:
            _DEAL["question_bar"] = float(got["question_bar"])
        _DEAL["links"].extend(got.get("links") or [])
        _DEAL["deal"].update(got.get("deal") or {})


def full_read() -> dict[str, Any]:
    """This compile's readings beyond types: ``{"reads", "links", "deal"}``."""
    return {k: _DEAL.get(k) for k in ("reads", "links", "deal", "questions", "question_bar")}


def set_deal(documents: list[dict[str, Any]] | None, artifact_sha256: dict[str, str] | None = None) -> None:
    _DEAL.clear()
    _DEAL.update(documents=list(documents or []), sha=dict(artifact_sha256 or {}), types={}, docs={},
                 reads={}, links=[], deal={}, questions=[])


def _delivered_at(sha256: str) -> Any:
    try:
        from app.core.document_lifecycle.dataset import lookup

        rec = lookup(sha256) or {}
    except Exception:
        return None
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
    one before it, as on the labeling page). Kind is the atoms' event kind,
    else the file's type by its bytes (``_file_kind``)."""
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
    first: dict[str, Any] = {}
    event: dict[str, str] = {}
    for a in atoms:
        d = _doc_id(a)
        first.setdefault(d, a)
        if d not in event or not event[d]:
            event[d] = _event_kind(a)
    sha = {k: str(v.get("sha256") or "") for k, v in (fingerprints or {}).items() if v.get("sha256")}
    docs = []
    for pos, (d, fp) in enumerate((fingerprints or {}).items()):
        name = str(fp.get("filename") or "")
        p = prov.get(name) or {}
        authored = p.get("authored_at") or (index.get(name) or {}).get("authored_at") or None
        cb = p.get("carried_by") if isinstance(p.get("carried_by"), dict) else {}
        carrier = prov.get(str(cb.get("filename") or "")) or {}
        sent = (cb.get("sent_at") or carrier.get("authored_at")) if cb else None
        if not sent and sha.get(d):
            # doc_origin.origin_for's last rule: the message the lifecycle
            # matched as delivering the file (live 000002: the SOW draft
            # arrived with the note at 23:32:57.777, as the page had it).
            sent = _delivered_at(sha[d])
        kind = event.get(d) or (_file_kind(first[d], sha) if d in first else "file")
        docs.append({"doc_id": d, "kind": kind, "sent_at": sent, "authored_at": authored,
                     "filename": name, "pos": pos})
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


# ------------------------------------------- SOWSMITH_C3_FALLBACK=never

class C3HeadsUnavailable(RuntimeError):
    """The heads could not answer and ``SOWSMITH_C3_FALLBACK=never`` forbids
    handing their lines to the LLM instead: the compile fails, saying why.
    Every stage that catches its own errors re-raises this one."""


def fallback_mode() -> str:
    """``llm`` (default: service trouble sends every atom to the LLM) or
    ``never`` (wait for the heads; past the ceiling, fail the compile)."""
    v = os.environ.get("SOWSMITH_C3_FALLBACK", "").strip().lower()
    return "never" if v == "never" else "llm"


#: Set in the shadow read's own thread: shadow decides nothing, so it keeps
#: the ``llm`` behaviour (abstain on trouble) in either mode.
_SHADOW = threading.local()


def _never() -> bool:
    return fallback_mode() == "never" and not getattr(_SHADOW, "on", False)


#: Answers worth waiting through: the request timed out (408), was rate
#: limited (429), or the service is starting, overloaded or failing (5xx).
#: Any other 4xx is the request's own fault and is not retried.
_RETRY_4XX = (408, 429)
_sleep = time.sleep  # tests replace it


class _NotReady(Exception):
    """/health answered but the heads are still loading."""


def _retryable(exc: BaseException) -> tuple[bool, str]:
    """(worth retrying, reason) for one failed attempt."""
    import socket
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        code = int(exc.code or 0)
        return (code >= 500 or code in _RETRY_4XX), f"HTTP {code}"
    if isinstance(exc, _NotReady):
        return True, f"warming: {exc}"
    if isinstance(exc, (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError)):
        return True, f"{type(exc).__name__}: {exc}"
    return False, f"{type(exc).__name__}: {exc}"


def _say_waiting(text: str) -> None:
    """Name the wait on the running stage (the compile heartbeat's ``item``),
    so a held compile reads as waiting for the heads, not as wedged."""
    try:
        from app.core import telemetry

        telemetry.set_stage_item(text)
    except Exception:
        pass


def _wait_for_heads(path: str, body: dict[str, Any] | None, status: dict[str, Any], what: str
                    ) -> dict[str, Any]:
    """``never`` mode: ask /health, then ``path``, until the service answers.

    Retries unreachable / warming / 408 / 429 / 5xx / timeouts with backoff,
    reading with a generous timeout, until ``SOWSMITH_C3_WAIT_MAX_S``; then,
    or at once on any other failure, raises ``C3HeadsUnavailable``. Never
    returns an empty answer for service trouble."""
    ceiling = _env_float("SOWSMITH_C3_WAIT_MAX_S", 1800.0)
    delay = max(0.0, _env_float("SOWSMITH_C3_RETRY_S", 5.0))
    delay_max = max(delay, _env_float("SOWSMITH_C3_RETRY_MAX_S", 60.0))
    connect = _env_float("SOWSMITH_C3_CONNECT_S", 5.0)
    read_s = max(_env_float("SOWSMITH_C3_BUDGET_S", 90.0) + 10.0,
                 _env_float("SOWSMITH_C3_NEVER_READ_S", 600.0))
    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            health = _request("/health", None, timeout=connect)
            if not health.get("ready"):
                raise _NotReady(str(health.get("error") or "still loading")[:200])
            got = _request(path, body, timeout=read_s)
            if attempt > 1:
                log.info("C3 heads answered %s after %d attempts (%.0f s)", what, attempt,
                         time.monotonic() - started)
                _say_waiting("")
            status.pop("waiting", None)
            return got
        except Exception as exc:
            retry, reason = _retryable(exc)
            waited = time.monotonic() - started
            if not retry:
                status.update(state="read_failed", reason=reason[:300], remote=True)
                _say_waiting("")
                raise C3HeadsUnavailable(
                    f"C3 heads refused the {what} ({reason}); SOWSMITH_C3_FALLBACK=never "
                    "does not hand the lines to the LLM") from exc
            if waited >= ceiling:
                status.update(state="unavailable", reason=reason[:300], remote=True,
                              waiting={"attempts": attempt, "waited_s": round(waited, 1)})
                _say_waiting("")
                raise C3HeadsUnavailable(
                    f"C3 heads unavailable for the {what} after {waited:.0f} s and {attempt} attempts "
                    f"(last: {reason}); SOWSMITH_C3_FALLBACK=never fails the compile instead of "
                    "handing the lines to the LLM") from exc
            pause = min(delay, max(0.0, ceiling - waited))
            status.update(state="waiting", reason=reason[:300], remote=True,
                          waiting={"attempts": attempt, "waited_s": round(waited, 1),
                                   "ceiling_s": ceiling})
            log.warning("C3 heads not answering the %s (%s); attempt %d, waited %.0f of %.0f s, "
                        "retrying in %.0f s", what, reason, attempt, waited, ceiling, pause)
            _say_waiting(f"waiting for C3 heads ({what}): {reason[:80]}; "
                         f"{waited:.0f}/{ceiling:.0f} s, attempt {attempt}")
            _sleep(pause)
            delay = min(delay * 2 if delay else delay_max, delay_max)


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
    body = {"deal": {"deal_id": deal_id, "deal_name": deal.get("deal_name") or "", "account": deal.get("account_name") or "",
                     "atoms": rows, "site_count": 0,
                     "c3_brief": {"sections": {k: [{"id": x.get("id")} for x in (v or []) if isinstance(x, dict)]
                                               for k, v in (brief.get("sections") or {}).items()},
                                  "job_details": brief.get("job_details") or []}}}
    if _never():
        return _wait_for_heads("/v1/quote", body, {}, "quote read")
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
         "speaker_role": a.speaker_role, "speaker_side": a.speaker_side, "atom_id": a.atom_id}
        for a in deal.atoms]}


#: The base's name for a type the parser calls by another. ``work_scope_item``
#: is the parser's ``scope_item`` (the labelers' word for it); without this
#: every sure work_scope_item answer had no parser type and went to the LLM.
BASE_ALIAS = {"work_scope_item": "scope_item"}


def _aliased(types: dict[str, Any] | None) -> dict[str, Any]:
    return {k: ([BASE_ALIAS.get(str(v[0]), v[0]), *list(v[1:])] if v else v) for k, v in (types or {}).items()}


def _remote(deal_id: str, rows: list[dict[str, Any]], keys: list[str], ids: list[int]
            ) -> dict[int, tuple[str, float]]:
    import urllib.error

    budget = _env_float("SOWSMITH_C3_BUDGET_S", 90.0)
    started = time.monotonic()
    shared = _DEAL.get("types") or {}
    if keys and all(k in shared for k in keys):
        # This compile already read the deal (document_job_scope, on the same
        # record): its lines only dropped since, so the readings stand.
        _STATUS.update(state="read", reason="shared", remote=True, lines=len(rows), read_ms=0.0)
        return {i: (str(shared[k][0]), float(shared[k][1] or 0.0)) for i, k in zip(ids, keys)
                if shared[k] and shared[k][0] is not None}
    if _never():
        payload = {"deal_id": deal_id, "company": "purtera", "atoms": rows}
        got = _wait_for_heads("/read", _brief_body({"deal": payload, "budget_s": budget}),
                              _STATUS, "type read")
        _keep_full(got)
        return _remote_types(got, rows, keys, ids, started)
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
    return _remote_types(got, rows, keys, ids, started)


def _remote_types(got: dict[str, Any], rows: list[dict[str, Any]], keys: list[str], ids: list[int],
                  started: float) -> dict[int, tuple[str, float]]:
    _STATUS.update(state="read", reason="", remote=True, lines=len(rows),
                   read_ms=round((time.monotonic() - started) * 1000, 1),
                   service_read_ms=got.get("read_ms"))
    types = _aliased(got.get("types"))
    if "types" in _DEAL:
        _DEAL["types"].update(types)
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
        if _never():
            raise C3HeadsUnavailable(f"C3 heads model not loaded ({_STATUS.get('reason') or _STATUS.get('state')}); "
                                     "SOWSMITH_C3_FALLBACK=never fails the compile")
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
        if _never():
            raise C3HeadsUnavailable(f"C3 type read passed its {budget:g} s budget ({exc}); "
                                     "SOWSMITH_C3_FALLBACK=never fails the compile") from exc
        log.warning("C3 type base read stopped (%s); every atom goes to the LLM", exc)
        return {}
    except Exception as exc:
        _STATUS.update(state="read_failed", reason=f"{type(exc).__name__}: {exc}"[:300],
                       lines=len(ids), read_ms=round((time.monotonic() - started) * 1000, 1))
        if _never():
            raise C3HeadsUnavailable(f"C3 type read failed ({type(exc).__name__}: {exc}); "
                                     "SOWSMITH_C3_FALLBACK=never fails the compile") from exc
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
            _SHADOW.on = True  # shadow decides nothing: abstain on trouble, never wait or raise
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
    base, takes the base's type where the base is sure (at least 0.85 and its
    own bar) and both types are simple. Validation deal 010043 (2026-10-15):
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
        if p < max(0.85, sure_for(ans)):
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
    raw = os.environ.get("SOWSMITH_C3_AUTHORITY_SURE_BY_TYPE", "")
    for part in raw.split(","):
        k, _, v = part.partition(":")
        if k.strip() == answer:
            try:
                return float(v)
            except ValueError:
                pass
    return _env_float("SOWSMITH_C3_AUTHORITY_SURE", 0.9)


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
    """Read the deal again when lines were created after the base's read
    (signature splits, site backfills, geo mentions): 97 of 010038's labeled
    lines had no reading at all, so no head ever judged them. The read is of
    the whole deal (a line is read in its document's context) and replaces the
    first one. Returns the lines that were new."""
    if not authority_enabled() or "types" not in _DEAL:
        return 0
    seen = _DEAL.get("types") or {}
    sha = _DEAL.get("sha")
    new = 0
    for a in atoms:
        if str(getattr(a, "id", "") or "") in seen:
            continue
        try:
            if skipped(a, sha):
                continue
        except Exception:
            pass
        new += 1
    if not new:
        return 0
    # A failed re-read must not cost the first read: keep it, and take the
    # new one only when it came back (010038 lost every base reading when a
    # re-read timed out, 2026-10-17).
    kept = {k: _DEAL.get(k) for k in ("docs", "reads", "links", "deal", "questions", "question_bar", "types")}
    _DEAL.update(docs={}, reads={}, links=[], deal={}, questions=[])
    _DEAL["types"] = dict(kept["types"] or {})
    documents(atoms)
    if not _DEAL.get("reads"):
        _DEAL.update({k: v for k, v in kept.items() if v is not None})
        return 0
    return new


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
        # At the end the base's answer wins for every type that needs no
        # values, locked or not: a later stage rebuilds a line's value and
        # loses its lock (010043's "Do you have the model of the camera?" was
        # read answered_question at 0.9985 and ended open_question, so the
        # brief made it a customer responsibility).
        if final and not is_locked(a) and ans not in _VALUELESS:
            continue
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
    if _DEAL.get("docs"):
        return dict(_DEAL["docs"])
    import urllib.error

    started = time.monotonic()
    budget = _env_float("SOWSMITH_C3_BUDGET_S", 90.0)
    deal_id, rows, _keys = _rows(atoms, artifact_sha256)
    try:
        if _url() and _never():
            payload = {"deal_id": deal_id, "company": "purtera", "atoms": rows}
            body = _brief_body({"deal": payload, "budget_s": budget, "document_heads": list(keys)})
            got = _wait_for_heads("/read", body, _DOC_STATUS, "document read")
            raw = got.get("documents") or {}
            if "types" in _DEAL:
                _DEAL["types"].update(_aliased(got.get("types")))
            _keep_full(got)
        elif _url():
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
                _DEAL["types"].update(_aliased(got.get("types")))
            _keep_full(got)
        else:
            from ml.c3.predict import read_deal

            loaded = get_model()
            if loaded is None:
                _DOC_STATUS.update(state="no_model", reason="", read_ms=0.0)
                if _never():
                    raise C3HeadsUnavailable("C3 heads model not loaded for the document read; "
                                             "SOWSMITH_C3_FALLBACK=never fails the compile")
                return {}
            model, schema, device = loaded
            deal, _ = _build(atoms, artifact_sha256)
            doc = read_deal(model, deal, schema, max_atoms=MAX_ATOMS, device=device, lines_only=True,
                            deadline=started + budget, heads=("col:label_type",), document_heads=keys)
            raw = {d: {k: [v["answer"], v.get("p")] for k, v in h.items()}
                   for d, h in (doc.get("documents") or {}).items()}
    except C3HeadsUnavailable:
        raise
    except Exception as exc:
        if _never():
            _DOC_STATUS.update(state="over_budget" if isinstance(exc, TimeoutError) else "read_failed",
                               reason=f"{type(exc).__name__}: {exc}"[:300],
                               read_ms=round((time.monotonic() - started) * 1000, 1))
            raise C3HeadsUnavailable(f"C3 document read failed ({type(exc).__name__}: {exc}); "
                                     "SOWSMITH_C3_FALLBACK=never fails the compile") from exc
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
    _DOC_STATUS.update(state="read", reason="", documents=len(out),
                       read_ms=round((time.monotonic() - started) * 1000, 1))
    return out
