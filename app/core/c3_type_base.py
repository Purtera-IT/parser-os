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

#: The window the base was trained and scored with (ml/c3/pod/*.sh ATOMS):
#: a deal past this is read in windows. Reading smaller windows than it saw
#: in training changes its answers.
MAX_ATOMS = 8192


# What the layer last did, for the compile's ML manifest and the nightly
# report: state is off | loaded | load_failed | no_memory | over_budget |
# read_failed | read. Counts and times only.
_STATUS: dict[str, Any] = {"state": "off"}


def status() -> dict[str, Any]:
    return dict(_STATUS)


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


def acts_on() -> frozenset[str] | None:
    """The answers the layer may act on (``SOWSMITH_C3_TYPES``, comma
    separated, ``_keep`` included by name), or None for every value-light
    type and ``_keep``. Promotion widens this one type at a time, each only
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
    "vendor_datasheet", "floor_plan", "case_manifest",
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
    return "file"


def doc_kind(atom: Any, artifact_sha256: dict[str, str] | None = None) -> str:
    """The training files' ``doc_kind`` for this atom's document.

    ``artifact_sha256``: artifact id -> the file's sha256 (the compile
    manifest's fingerprints); without it a file falls back to its name."""
    return _event_kind(atom) or _file_kind(atom, artifact_sha256)


def skipped(atom: Any, artifact_sha256: dict[str, str] | None = None) -> bool:
    from ml.c3.data import is_deal_kit

    kind = doc_kind(atom, artifact_sha256)
    return kind in _SKIP_KINDS or is_deal_kit(kind)


def _section(atom: Any) -> str:
    loc = _locator(atom)
    sp = loc.get("section_path")
    if isinstance(sp, (list, tuple)) and sp:
        return " > ".join(str(x) for x in sp if str(x).strip())
    if isinstance(sp, str) and sp.strip():
        return sp
    # A sheet is a workbook row's section, as the classifier reads it.
    sheet = loc.get("sheet")
    return str(sheet) if sheet else ""


#: The speaker values training saw; anything else is '' (as on ~95% of lines).
_ROLES = frozenset({"internal", "reseller", "unknown"})
_SIDES = frozenset({"ours", "theirs"})


def _speaker(atom: Any) -> tuple[str, str]:
    """(role, side): the page's ``saidBy``. An email or pasted-note line carries
    its party (deal_parties); a transcript line only the speaker's role. A
    value training never saw is '', like a line that speaks for nobody."""
    by = _value(atom).get("said_by")
    if isinstance(by, dict):
        role, side = str(by.get("role") or by.get("role_guess") or ""), str(by.get("side") or "")
    else:
        role, side = str(_locator(atom).get("speaker_role") or ""), ""
    role, side = role.strip().lower(), side.strip().lower()
    return (role if role in _ROLES else ""), (side if side in _SIDES else "")


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
    from ml.c3.data import Atom, DealExample

    times: list[float | None] = [_own_time(a) for a in atoms]
    doc_ids = [_doc_id(a) for a in atoms]
    kinds: dict[str, str] = {}
    for a, d in zip(atoms, doc_ids):
        if not kinds.get(d):
            kinds[d] = _event_kind(a)
    for a, d in zip(atoms, doc_ids):
        if not kinds[d]:
            kinds[d] = _file_kind(a, artifact_sha256)
    # A document's arrival is its earliest line's own time, filled forward in
    # the order documents first appear in the compile.
    first: dict[str, float] = {}
    order: list[str] = []
    for d, t in zip(doc_ids, times):
        if d not in order:
            order.append(d)
        if t is not None and (d not in first or t < first[d]):
            first[d] = t
    doc_time: dict[str, float] = {}
    prev = next((first[d] for d in order if d in first), None)
    for d in order:
        prev = first.get(d, prev)
        if prev is not None:
            doc_time[d] = prev
    known = [t for t in times if t is not None] + list(doc_time.values())
    latest = max(known) if known else 0.0

    rows, keys, seen = [], [], set()
    for i, (a, d, t) in enumerate(zip(atoms, doc_ids, times)):
        key = _key(a, i)
        if key in seen:
            key = f"{key}#{i}"
        seen.add(key)
        keys.append(key)
        role, side = _speaker(a)
        rows.append(Atom(
            key=key, text=str(getattr(a, "raw_text", "") or ""),
            entered_at=t if t is not None else doc_time.get(d, latest),
            doc_id=d, doc_kind=kinds[d], section=_section(a), order=float(i),
            speaker_role=role, speaker_side=side, atom_id=str(getattr(a, "id", "") or "")))
    # Time order is the deal's order: ties keep compile order (DealExample.from_dict).
    rows.sort(key=lambda r: (r.entered_at, r.order))
    deal_id = next((str(getattr(a, "project_id", "") or "") for a in atoms
                    if getattr(a, "project_id", None)), "") or "live"
    return DealExample(deal_id=deal_id, atoms=rows, company="purtera"), keys


def _read(atoms: list[Any], artifact_sha256: dict[str, str] | None = None
          ) -> dict[int, tuple[str, float]]:
    from ml.c3.predict import read_deal

    loaded = get_model()
    if loaded is None:
        return {}
    model, schema, device = loaded
    deal, keys = _build(atoms, artifact_sha256)
    started = time.monotonic()
    budget = _env_float("SOWSMITH_C3_BUDGET_S", 90.0)
    try:
        doc = read_deal(model, deal, schema, max_atoms=MAX_ATOMS, device=device,
                        lines_only=True, deadline=started + budget)
    except TimeoutError as exc:
        _STATUS.update(state="over_budget", reason=str(exc)[:300], lines=len(atoms),
                       read_ms=round((time.monotonic() - started) * 1000, 1))
        log.warning("C3 type base read stopped (%s); every atom goes to the LLM", exc)
        return {}
    except Exception as exc:
        _STATUS.update(state="read_failed", reason=f"{type(exc).__name__}: {exc}"[:300],
                       lines=len(atoms), read_ms=round((time.monotonic() - started) * 1000, 1))
        raise
    _STATUS.update(state="read", reason="", lines=len(atoms),
                   read_ms=round((time.monotonic() - started) * 1000, 1))
    lines = doc.get("lines") or {}
    out: dict[int, tuple[str, float]] = {}
    for a, key in zip(atoms, keys):
        got = ((lines.get(key) or {}).get("reads") or {}).get("col:label_type")
        if got and got.get("answer") is not None:
            out[id(a)] = (str(got["answer"]), float(got.get("p") or 0.0))
    return out


def classify(atoms: list[Any], artifact_sha256: dict[str, str] | None = None
             ) -> dict[int, tuple[str, float]]:
    """``{id(atom): (label_type answer, probability)}`` for every atom read.

    Reads the whole deal once. Empty when the model is not loaded."""
    return _read(atoms, artifact_sha256)


# ----------------------------------------------------------------- shadow

_LAST_SHADOW: dict[str, Any] = {}


def get_last_shadow() -> dict[str, Any]:
    return dict(_LAST_SHADOW)


def report_shadow(would: dict[int, tuple[Any, str, str, float, str]]) -> dict[str, Any]:
    """What the layer would have done beside what the compile did.

    ``would``: id(atom) -> (atom, type before, answer, p, "assign" | "keep").
    An assignment agrees when the atom ended as that type; a keep when it
    ended as the type it had. No text leaves the compile."""
    def _t(a: Any) -> str:
        t = getattr(a, "atom_type", None)
        return str(getattr(t, "value", t) or "")

    agree = disagree = 0
    by_answer: dict[str, dict[str, int]] = {}
    # What the compile ended with where the layer would have said otherwise,
    # as "layer answer->compile type": the dev review reads these pairs.
    pairs: dict[str, int] = {}
    for a, before, answer, _p, act in would.values():
        want = answer if act == "assign" else before
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
