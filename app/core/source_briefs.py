"""Per-source briefs: what each source of a deal tells the person quoting it.

C1 of purpulse-frontend ``docs/design/sources-reader/CALM-PLAN.md``. The Sources
tab shows one source at a time (a call, a file, an email thread, the deal's
notes). The deal brief (``c3_brief``) is written across every source and
cannot answer "what did THIS call say about the quote". So each source gets its
own short brief:

* ``lead`` -- one or two sentences on what the source is and what it adds;
* ``sections`` -- bullets under a fixed set of headings (``HEADINGS``);
* ``inputs`` -- label/value pairs a quote form would take from it, ``open``
  when the source raises a point and leaves it unanswered.

Every sentence, bullet and input cites the ids of the atoms it was written
from, and anything citing an id this source did not send is dropped and
counted (the ``c3_brief.write`` rule). The ids are the envelope's atom ids, so
the reader turns a citation into "6:18" or "p. 3" with the locator it already
has.

Off unless ``SOWSMITH_SOURCE_BRIEFS`` is on. One LLM call per source, in
parallel, through ``c3_brief._complete`` (retries an empty or unparseable
reply). A source whose lines have not changed is not written again: the result
is cached under a hash of the prompt version, the model and the lines sent.
The cache is the parser's local payload cache, so it lasts as long as the
worker's disk does.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any

log = logging.getLogger(__name__)

#: Bumped whenever the prompt or the lines sent change: it is part of the cache key.
VERSION = "sb1"

#: The only headings a brief may use, in this order. The reader shows them as written.
HEADINGS: tuple[str, ...] = (
    "Scope",
    "Labor and method",
    "Sites and schedule",
    "Reporting",
    "Commercial",
    "People",
    "Open questions",
)
ITEMS_PER_SECTION = 5
MAX_INPUTS = 10
LABEL_CHARS = 32
VALUE_CHARS = 120
LINE_CHARS = 400

#: A source with fewer lines than this has nothing to brief (a one-line status note).
MIN_LINES = int(os.environ.get("SOWSMITH_SOURCE_BRIEF_MIN_LINES", "4"))
#: Lines sent per source. A 31-minute call reads ~390 atoms; the least sure go first when cut.
MAX_LINES = int(os.environ.get("SOWSMITH_SOURCE_BRIEF_MAX_LINES", "180"))
MAX_SOURCES = int(os.environ.get("SOWSMITH_SOURCE_BRIEF_MAX_SOURCES", "40"))
WORKERS = max(1, int(os.environ.get("SOWSMITH_SOURCE_BRIEF_WORKERS", "4")))
TOKENS = int(os.environ.get("SOWSMITH_SOURCE_BRIEF_TOKENS", "2500"))

#: An email header the parser kept as a line, and what is left of a signature:
#: the same two shapes the reader sets aside (purpulse-frontend factModel.isBoilerplate).
_HEADER_LINE = re.compile(r"^\s*(?:from|to|cc|bcc|sent|date|subject)\s*:", re.I)
_LINK_ONLY = re.compile(
    r"^\s*(?:(?:https?://)?[\w.-]+\.(?:com|net|org|io|co|us|ai)(?:/\S*)?|[\w.+-]+@[\w.-]+\.\w+|[+()\d][\d ().-]{6,})\s*$",
    re.I,
)
#: The metadata line a HubSpot note export starts with: "note_id=… | author=… | date=…".
_META_LINE = re.compile(r"^\s*\w+=[^|]*\|\s*\w+=", re.I)
_ID = re.compile(r"[\[\]\s]")


def enabled() -> bool:
    return os.environ.get("SOWSMITH_SOURCE_BRIEFS", "").strip().lower() in ("1", "true", "yes", "on")


def _model() -> str | None:
    return os.environ.get("SOWSMITH_SOURCE_BRIEF_MODEL") or os.environ.get("SOWSMITH_C3_BRIEF_MODEL") or None


# ── which sources ────────────────────────────────────────────────────────────


def _kind_of(doc: dict[str, Any]) -> str:
    name = str(doc.get("filename") or "").lower()
    kind = str(doc.get("artifact_type") or "").lower()
    if "-hs-note-" in name:
        return "note"
    if kind == "transcript" or name.endswith("-transcript.json") or "-fireflies-" in name:
        return "call"
    if kind in ("email", "msg", "mbox") or name.endswith((".eml", ".msg")):
        return "email"
    return "file"


def _title_of(doc: dict[str, Any]) -> str:
    """The filename without the deal number and the export's machine id."""
    name = str(doc.get("filename") or "").rsplit("/", 1)[-1]
    name = re.sub(r"\.(json|txt|eml|msg|pdf|docx?|xlsx?|csv|pptx?)$", "", name, flags=re.I)
    name = re.sub(r"^\d{4,}-(?:fireflies-[0-9A-Z]+-|hs-(?:note|email|meeting)-\d+-?)", "", name)
    name = re.sub(r"-transcript$", "", name)
    return re.sub(r"[-_]+", " ", name).strip() or str(doc.get("filename") or "")


def sources_of(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """The deal's sources as the reader lists them: a call or a file by its
    artifact, an email thread by its thread id, and every note together (the
    reader shows notes as one feed)."""
    docs = [d for d in envelope.get("documents") or [] if isinstance(d, dict) and d.get("artifact_id")]
    by_id = {str(d["artifact_id"]): d for d in docs}
    out: list[dict[str, Any]] = []
    in_thread: set[str] = set()
    for t in envelope.get("email_threads") or []:
        ids = [str(i) for i in t.get("artifact_ids") or [] if str(i) in by_id]
        if not ids:
            continue
        in_thread.update(ids)
        out.append({"key": f"thread:{t.get('thread_id')}", "kind": "email",
                    "title": str(t.get("name") or "Email thread"), "artifact_ids": ids})
    notes: list[str] = []
    for d in docs:
        aid = str(d["artifact_id"])
        if aid in in_thread:
            continue
        kind = _kind_of(d)
        if kind == "note":
            notes.append(aid)
        else:
            out.append({"key": aid, "kind": kind, "title": _title_of(d), "artifact_ids": [aid]})
    if notes:
        notes.sort(key=lambda a: str(by_id[a].get("authored_at") or ""))
        out.append({"key": "notes", "kind": "note", "title": "Notes on the deal", "artifact_ids": notes})
    return out


# ── which lines ──────────────────────────────────────────────────────────────


def _boilerplate(atom: dict[str, Any]) -> bool:
    from app.core.c3_brief import NEVER

    if str(atom.get("atom_type") or "") in NEVER:
        return True
    s = atom.get("structured") if isinstance(atom.get("structured"), dict) else {}
    if s.get("chatter") is True or isinstance(s.get("admission_regex"), str):
        return True
    if any(f in ("chatter", "admission_regex") for f in atom.get("review_flags") or []):
        return True
    text = str(atom.get("text") or "")
    return bool(_HEADER_LINE.match(text) or _LINK_ONLY.match(text) or _META_LINE.match(text))


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def where(atom: dict[str, Any]) -> str:
    """Where a line sits, in the reader's words: "6:18", "p. 3", "message 2"."""
    loc = atom.get("locator") if isinstance(atom.get("locator"), dict) else {}
    secs = _num(loc.get("timestamp_start"))
    if secs is None and _num(loc.get("start_ms")) is not None:
        secs = _num(loc.get("start_ms")) / 1000
    if secs is not None:
        s = max(0, round(secs))
        return f"{s // 60}:{s % 60:02d}"
    page = _num(loc.get("page"))
    if page is not None:
        return f"p. {int(page) + 1}"
    msg = _num(loc.get("message_index"))
    if msg is not None:
        return f"message {int(msg) + 1}"
    return ""


def lines_of(envelope: dict[str, Any], source: dict[str, Any]) -> list[dict[str, str]]:
    """The source's lines a quote could use, in reading order: boilerplate and
    repeats out, the least sure cut first when there are too many."""
    ids = set(source["artifact_ids"])
    dated = source["kind"] == "note"
    when = {str(d.get("artifact_id")): str(d.get("authored_at") or "")[:10]
            for d in envelope.get("documents") or [] if isinstance(d, dict)}
    rows: list[tuple[int, float, dict[str, str]]] = []
    seen: set[str] = set()
    for i, a in enumerate(envelope.get("atoms") or []):
        if not isinstance(a, dict) or str(a.get("artifact_id")) not in ids or not a.get("id") or _boilerplate(a):
            continue
        text = re.sub(r"\s+", " ", str(a.get("text") or "")).strip()
        fold = text.lower()
        if not text or fold in seen:
            continue
        seen.add(fold)
        w = where(a)
        if dated and when.get(str(a.get("artifact_id"))):
            w = when[str(a.get("artifact_id"))]
        conf = _num(a.get("confidence"))
        rows.append((i, conf if conf is not None else 0.5, {"id": str(a["id"]), "where": w, "text": text[:LINE_CHARS]}))
    if len(rows) > MAX_LINES:
        keep = {r[0] for r in sorted(rows, key=lambda r: -r[1])[:MAX_LINES]}
        rows = [r for r in rows if r[0] in keep]
    return [r[2] for r in rows]


# ── writing ──────────────────────────────────────────────────────────────────

_KIND_WORDS = {
    "call": "a call transcript (speech is transcribed and messy; times are m:ss into the call)",
    "file": "a document (p. N is its page)",
    "email": "an email thread",
    "note": "the deal's notes, oldest first, each line with the day it was written",
}

_PROMPT = (
    "You help a solutions architect at PurTera price a field-services job. PurTera is the provider asked to "
    "quote and do the work. Below are the lines a parser read from ONE source of the deal: %s. Write what this "
    "source tells the architect for the quote.\n"
    "LEAD: one or two sentences: what the source is and what it adds to the quote.\n"
    "SECTIONS: use only these headings, in this order, and leave out a heading with nothing under it: "
    + "; ".join(HEADINGS) + ". At most " + str(ITEMS_PER_SECTION) + " short bullets each. A bullet states one thing "
    "the price depends on: the work, counts, equipment and part numbers, method and time per unit, places, dates "
    "and windows, constraints, who does or supplies what, paperwork. Leave out greetings, small talk, talk about "
    "the meeting itself, and anything about PurTera's own company or capabilities. Open questions: what this "
    "source asks or leaves open that changes the price, one question each.\n"
    "INPUTS: at most " + str(MAX_INPUTS) + " label/value pairs a quote form would take from this source. A label is "
    "one to three words (Kit, Trailer type, Install time, Pilot, Batch size); a value is under 12 words. "
    "open=true when the source raises the point and does not answer it; the value then says what is unknown.\n"
    "RULES: every lead sentence, bullet and input cites the [ids] of the lines it uses and says nothing those lines "
    "do not say. Copy counts, part numbers, sizes and dates exactly. When lines disagree, give both and say so. A "
    "later line replaces an earlier one on the same point. Write plainly in the third person; never quote filler "
    "words.\n"
    "Return JSON only: {\"lead\": {\"text\": str, \"cites\": [id]}, \"sections\": [{\"heading\": str, \"items\": "
    "[{\"text\": str, \"cites\": [id]}]}], \"inputs\": [{\"label\": str, \"value\": str, \"open\": bool, "
    "\"cites\": [id]}]}\n\n"
    "SOURCE: %s\nLINES: %s"
)


def prompt_for(source: dict[str, Any], lines: list[dict[str, str]]) -> str:
    body = [f"[{x['id']}] " + (f"({x['where']}) " if x["where"] else "") + x["text"] for x in lines]
    return _PROMPT % (_KIND_WORDS.get(source["kind"], "a source"), source["title"], json.dumps(body))


def _cache_key(prompt: str, model: str | None) -> str:
    h = hashlib.sha256()
    for part in (VERSION, model or "", prompt):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return "source_brief_" + h.hexdigest()[:40]


def _cites(raw: Any, allowed: set[str]) -> list[str] | None:
    """The cited ids, or None when any of them is not a line this source sent."""
    if isinstance(raw, str):
        raw = re.split(r"[,\s]+", raw)
    got = [_ID.sub("", str(c)) for c in raw or [] if _ID.sub("", str(c))]
    if not got or any(c not in allowed for c in got):
        return None
    return list(dict.fromkeys(got))


def check(doc: dict[str, Any], lines: list[dict[str, str]]) -> dict[str, Any]:
    """What survives of the writer's answer: only text whose every citation is
    a line of this source, under a known heading, within the caps."""
    allowed = {x["id"] for x in lines}
    where_of = {x["id"]: x["where"] for x in lines}
    dropped = {"uncited": 0, "heading": 0}

    def place(cites: list[str]) -> list[str]:
        return list(dict.fromkeys(where_of[c] for c in cites if where_of.get(c)))

    lead = None
    raw_lead = doc.get("lead")
    if isinstance(raw_lead, dict) and str(raw_lead.get("text") or "").strip():
        c = _cites(raw_lead.get("cites"), allowed)
        if c is None:
            dropped["uncited"] += 1
        else:
            lead = {"text": str(raw_lead["text"]).strip(), "cites": c}
    sections: dict[str, list[dict[str, Any]]] = {}
    for sec in doc.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        heading = next((h for h in HEADINGS if h.lower() == str(sec.get("heading") or "").strip().lower()), None)
        items = [x for x in sec.get("items") or [] if isinstance(x, dict) and str(x.get("text") or "").strip()]
        if heading is None:
            dropped["heading"] += len(items)
            continue
        for x in items:
            c = _cites(x.get("cites"), allowed)
            if c is None:
                dropped["uncited"] += 1
                continue
            bucket = sections.setdefault(heading, [])
            if len(bucket) < ITEMS_PER_SECTION:
                bucket.append({"text": str(x["text"]).strip(), "cites": c, "where": place(c)})
    inputs: list[dict[str, Any]] = []
    for x in doc.get("inputs") or []:
        if not isinstance(x, dict) or not str(x.get("label") or "").strip() or not str(x.get("value") or "").strip():
            continue
        c = _cites(x.get("cites"), allowed)
        if c is None:
            dropped["uncited"] += 1
            continue
        if len(inputs) < MAX_INPUTS:
            inputs.append({"label": str(x["label"]).strip()[:LABEL_CHARS], "value": str(x["value"]).strip()[:VALUE_CHARS],
                           "open": x.get("open") is True, "cites": c, "where": place(c)})
    return {
        "lead": lead,
        "sections": [{"heading": h, "items": sections[h]} for h in HEADINGS if sections.get(h)],
        "inputs": inputs,
        "dropped_uncited": dropped["uncited"],
        "dropped_heading": dropped["heading"],
    }


def _write_one(source: dict[str, Any], lines: list[dict[str, str]], model: str | None) -> dict[str, Any]:
    from app.core.c3_brief import _complete, _json_of
    from app.storage.repositories import load_cache_payload, save_cache_payload

    prompt = prompt_for(source, lines)
    key = _cache_key(prompt, model)
    base = {"key": source["key"], "kind": source["kind"], "title": source["title"],
            "artifact_ids": source["artifact_ids"], "lines_sent": len(lines), "input_hash": key}
    try:
        cached = load_cache_payload(key)
    except Exception:  # pragma: no cover - a broken cache never costs the brief
        cached = None
    if isinstance(cached, dict) and cached.get("version") == VERSION:
        return {**base, **{k: v for k, v in cached.items() if k != "version"}, "cached": True}
    raw = _complete(prompt, model, TOKENS)
    doc = _json_of(raw)
    if doc is None:
        return {**base, "lead": None, "sections": [], "inputs": [], "cached": False,
                "error": "the writer returned no usable JSON"}
    written = check(doc, lines)
    if written["lead"] or written["sections"] or written["inputs"]:
        try:
            save_cache_payload(key, {"version": VERSION, **written})
        except Exception:  # pragma: no cover
            log.info("source brief cache write failed for %s", source["key"])
    return {**base, **written, "cached": False}


def build(envelope: dict[str, Any]) -> dict[str, Any]:
    """Every source's brief, keyed by source. Sources with too little to say are
    listed under ``skipped`` with why, never silently left out."""
    from concurrent.futures import ThreadPoolExecutor

    started = time.monotonic()
    model = _model()
    todo, skipped = [], []
    for s in sources_of(envelope):
        lines = lines_of(envelope, s)
        if len(lines) < MIN_LINES:
            skipped.append({"key": s["key"], "why": f"{len(lines)} line(s) to read"})
        elif len(todo) >= MAX_SOURCES:
            skipped.append({"key": s["key"], "why": "over the per-deal source cap"})
        else:
            todo.append((s, lines))
    by_source: dict[str, Any] = {}
    if todo:
        with ThreadPoolExecutor(max_workers=min(WORKERS, len(todo))) as pool:
            futures = [(s, pool.submit(_write_one, s, lines, model)) for s, lines in todo]
            for s, f in futures:
                try:
                    by_source[s["key"]] = f.result()
                except Exception as exc:  # pragma: no cover - one source never costs the others
                    by_source[s["key"]] = {"key": s["key"], "kind": s["kind"], "title": s["title"],
                                           "artifact_ids": s["artifact_ids"], "lead": None, "sections": [],
                                           "inputs": [], "error": f"{type(exc).__name__}: {exc}"[:300]}
    briefs = list(by_source.values())
    return {
        "version": VERSION,
        "model": model or "default",
        "by_source": by_source,
        "skipped": skipped,
        "calls": sum(1 for b in briefs if not b.get("cached") and not b.get("error")),
        "cached": sum(1 for b in briefs if b.get("cached")),
        "errors": sum(1 for b in briefs if b.get("error")),
        "ms": round((time.monotonic() - started) * 1000, 1),
    }


def for_envelope(envelope: dict[str, Any]) -> dict[str, Any] | None:
    """``envelope["source_briefs"]`` when switched on, else None."""
    if not enabled():
        return None
    return build(envelope)
