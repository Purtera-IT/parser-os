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

#: What predict.py reads a deal with: a deal past this is read in windows.
MAX_ATOMS = 4096


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
    schema = load_schema()
    device = _device()
    model = live_student(schema, encoder).to(device)
    load_checkpoint(path, model, schema_sha256=digest([asdict(o) for o in schema.opportunities]))
    model.eval()
    return model, schema, device


def get_model() -> tuple[Any, Any, str] | None:
    """The process-wide model, or None when it cannot load (warned once)."""
    global _MODEL, _TRIED
    if _TRIED:
        return _MODEL
    with _LOCK:
        if not _TRIED:
            try:
                _MODEL = _load()
            except Exception as exc:
                _MODEL = None
                log.warning("C3 type base not loaded (%s: %s); the layer abstains",
                            type(exc).__name__, exc)
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


def doc_kind(atom: Any) -> str:
    """The page's ``docType``, as near as an atom can say it.

    The page takes it from the compiled document (lifecycle type, or the file
    kind); an atom carries only its source's artifact type, so that stands in,
    with the two kinds export.py names as events spelled its way: a transcript
    is a ``meeting``, a pasted HubSpot note a ``note``. A pricing workbook or
    an estimating calc is recognised by its file's name, so the layer can
    leave its lines alone."""
    name = re.sub(r"[^a-z]", "", _ref(atom, "filename").lower())
    if "dealkit" in name:
        return "deal_kit"
    if _CALC_NAME.search(name):
        return "estimating_calc"
    kind = str(_value(atom).get("kind") or "")
    if kind.startswith("hubspot_note"):
        return "note"
    art = (_ref(atom, "artifact_type") or "").strip().lower()
    if art == "transcript" or "utterance_index" in _locator(atom):
        return "meeting"
    return art


def skipped(atom: Any) -> bool:
    from ml.c3.data import is_deal_kit

    kind = doc_kind(atom)
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


def _speaker(atom: Any) -> tuple[str, str]:
    """(role, side): the page's ``saidBy``. An email or pasted-note line carries
    its party (deal_parties); a transcript line only the speaker's role. A line
    with neither speaks for nobody, as on the page."""
    by = _value(atom).get("said_by")
    if isinstance(by, dict):
        return str(by.get("role") or by.get("role_guess") or ""), str(by.get("side") or "")
    return str(_locator(atom).get("speaker_role") or ""), ""


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


def atoms_to_deal(atoms: list[Any]):
    """One ``DealExample`` from a compile's atoms (see ``_build``)."""
    return _build(atoms)[0]


def _build(atoms: list[Any]):
    """One ``DealExample`` from a compile's atoms, in arrival order.

    Field by field, against export.py:

    * key / atom_id: the atom's id (the page's ``labelKey`` is a label's key,
      not a model input; any stable key reads the same);
    * text: ``raw_text``;
    * doc_id: ``artifact_id``; doc_kind: ``doc_kind`` above;
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
    doc_ids = [str(getattr(a, "artifact_id", "") or _ref(a, "artifact_id")) for a in atoms]
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
            doc_id=d, doc_kind=doc_kind(a), section=_section(a), order=float(i),
            speaker_role=role, speaker_side=side, atom_id=str(getattr(a, "id", "") or "")))
    # Time order is the deal's order: ties keep compile order (DealExample.from_dict).
    rows.sort(key=lambda r: (r.entered_at, r.order))
    deal_id = next((str(getattr(a, "project_id", "") or "") for a in atoms
                    if getattr(a, "project_id", None)), "") or "live"
    return DealExample(deal_id=deal_id, atoms=rows, company="purtera"), keys


def _read(atoms: list[Any]) -> dict[int, tuple[str, float]]:
    from ml.c3.predict import read_deal

    loaded = get_model()
    if loaded is None:
        return {}
    model, schema, device = loaded
    deal, keys = _build(atoms)
    doc = read_deal(model, deal, schema, max_atoms=MAX_ATOMS, device=device)
    lines = doc.get("lines") or {}
    out: dict[int, tuple[str, float]] = {}
    for a, key in zip(atoms, keys):
        got = ((lines.get(key) or {}).get("reads") or {}).get("col:label_type")
        if got and got.get("answer") is not None:
            out[id(a)] = (str(got["answer"]), float(got.get("p") or 0.0))
    return out


def classify(atoms: list[Any]) -> dict[int, tuple[str, float]]:
    """``{id(atom): (label_type answer, probability)}`` for every atom read.

    Reads the whole deal once. Empty when the model is not loaded."""
    return _read(atoms)


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
    for a, before, answer, _p, act in would.values():
        want = answer if act == "assign" else before
        hit = _t(a) == want
        agree += hit
        disagree += not hit
        slot = by_answer.setdefault(answer, {"agree": 0, "disagree": 0})
        slot["agree" if hit else "disagree"] += 1
    payload = {
        "event": "c3_type_shadow", "stage": "typed_atom_classification", "sure": sure(),
        "would_assign": sum(1 for v in would.values() if v[4] == "assign"),
        "would_keep": sum(1 for v in would.values() if v[4] == "keep"),
        "agree": agree, "disagree": disagree, "by_answer": by_answer,
    }
    _LAST_SHADOW.clear()
    _LAST_SHADOW.update(payload)
    try:
        print(json.dumps(payload, ensure_ascii=True), file=sys.stderr)
    except Exception:
        pass
    return payload
