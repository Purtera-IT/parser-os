"""Does this document describe the job this deal is for?

A deal's attachments are whatever HubSpot associated with it, and on a
programme customer that includes other jobs. Live 010162 ("CDW- Sodexo SD-WAN
Program", 2026-09-15): a kiosk close-down at Delta Admin -- its packing list
("Delta Close Down.pdf") and a smart-hands thread ("RE: CDW Smart Hands SOW
Delta Admin 70598001") -- sat beside the SD-WAN scope. Nothing in either names
a deal number, so document scope (document_lifecycle/scope.py) read them as
this deal's, and the brief came out as a staff-augmentation job about kiosks,
badges and packing boxes, with its Deal Kit tasks drawn from the close-down.

The question is answered per CONVERSATION, through the decide() chokepoint:

    STORE (a PM taught "this thread is a different job")  ->  LLM  ->  this_deal

One conversation is one job. An email thread -- every message under one
subject, whatever reply marker or reference number someone typed in front of
it -- and the files that arrived with its messages are judged together, and
the verdict covers all of them. Measured on 010198 (2026-09-15): judged one
message at a time, three models called the "010179 POS Installation 8/2"
replies another job because the number differed from the deal's, when they
were the same customer confirming the same date; as one thread the same
models read them as this deal. And 010162's packing list, judged alone, is
counts and box sizes that say nothing about a job; beside the smart-hands
thread it arrived with, it is plainly the kiosk close-down.

The judge reads the deal's own name -- the seller's one-line statement of the
work -- against the conversation's subject and its first lines, and is asked
to compare the WORK, since customer, vendor, people and dates are shared by
every job the customer has. It is a hard discrimination: gpt-4.1-mini, the
worker's hosted teacher, called 000036's own request ("install their new
Samsung 65' display" on "San Fran TV mount") another job and 010162's kiosk
thread this deal; qwen3:32b on the same prompts was right on all seven deals
measured. So this decision names its model (``SOWSMITH_DOCUMENT_JOB_MODEL``,
default the local qwen3:32b) rather than taking the cheap default.

Only a confident ``other_job`` removes anything, and removal is lossless: the
atoms go to the suppression ledger like every other gate drop, so a PM can see
what was set aside and teach the opposite. Every verdict, kept or not, is
written to the compile trace. No store and no model means nothing moves.
"""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

RELATION = "document_job"
CANDIDATES = ["this_deal", "other_job"]
INSTRUCTION = (
    "The DEAL line names one job for one customer. Decide whether the DOCUMENT "
    "is about that same job (this_deal) or about a different piece of work for "
    "the same customer -- another work order, service call, site visit or "
    "project (other_job). Judge by the WORK described -- what is installed, "
    "removed, moved, serviced or priced -- never by the customer, vendor, "
    "people or dates, which one customer's jobs share. A document about the "
    "deal's work, its pricing, sites, schedule or contract is this_deal even if "
    "it also mentions other things. Answer other_job only when the document's "
    "work is clearly a different job from the one the DEAL line names; if you "
    "cannot tell, answer this_deal."
)
_DEFAULT_MODEL = "ollama:qwen3:32b"
_DEFAULT_TIMEOUT = 120
_DEFAULT_MIN_CONF = 0.8
_LINES = 14
_LINE_CHARS = 160
_META_KINDS = frozenset({"hubspot_note_meta", "email_header", "email_addressee"})
# A file mirrored from the CRM this close to a message that carried attachments
# arrived with that message, when the manifest gives no id to prove it.
_ARRIVED_WITH = timedelta(minutes=2)
# What people type in front of a subject that is not the subject: reply and
# forward markers, and reference tokens (a deal or ticket number, a bracketed
# tag of digits). Letters other than those markers are the subject itself.
_SUBJECT_PREFIX_RE = re.compile(r"^(?:(?:re|fw|fwd|aw|wg|sv|vs)\s*:\s*|[#\[\]()0-9._/-]+\s+)+")


def enabled() -> bool:
    return os.environ.get("SOWSMITH_DOCUMENT_JOB_SCOPE", "1").strip().lower() not in ("0", "false", "no", "off")


def judge_model() -> str | None:
    """The model this discrimination is routed to; '' or 'default' keeps decide()'s own."""
    raw = os.environ.get("SOWSMITH_DOCUMENT_JOB_MODEL", _DEFAULT_MODEL).strip()
    return None if raw.lower() in ("", "default") else raw


def judge_timeout() -> int:
    try:
        return max(5, int(os.environ.get("SOWSMITH_DOCUMENT_JOB_TIMEOUT", str(_DEFAULT_TIMEOUT))))
    except ValueError:
        return _DEFAULT_TIMEOUT


def min_confidence() -> float:
    try:
        return float(os.environ.get("SOWSMITH_DOCUMENT_JOB_MIN_CONF", str(_DEFAULT_MIN_CONF)))
    except ValueError:
        return _DEFAULT_MIN_CONF


def thread_key(subject: Any) -> str:
    """The conversation a subject line belongs to. "Re: 010198 Fw: POS
    Installation 8/2", "Fw: 010179 POS Installation 8/2" and "Re: POS
    Installation 8/2" are one thread: markers and reference numbers are what
    people type in front of a subject, not the subject."""
    s = " ".join(str(subject or "").lower().split())
    if not s:
        return ""
    return _SUBJECT_PREFIX_RE.sub("", s).strip() or s


def manifest_index(project_dir: Path | str | None) -> dict[str, dict[str, Any]]:
    """Per filename, what the manifest knows about where a document came from:
    the message subject, the attachment ids a message carried, a file's own
    external id, when it was authored, and its source."""
    if not project_dir:
        return {}
    path = Path(project_dir) / ".parser_manifest.json"
    if not path.is_file():
        return {}
    try:
        artifacts = json.loads(path.read_text(encoding="utf-8")).get("artifacts") or []
    except (OSError, ValueError, AttributeError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for a in artifacts:
        if not isinstance(a, dict) or not a.get("filename"):
            continue
        md = a.get("metadata") if isinstance(a.get("metadata"), dict) else {}
        out[str(a["filename"])] = {
            "subject": str(md.get("subject") or "").strip(),
            "attachment_ids": [str(x).strip() for x in (md.get("attachmentIds") or []) if str(x).strip()],
            "external_id": str(a.get("external_id") or "").strip(),
            "authored_at": str(md.get("timestamp") or a.get("authored_at") or "").strip(),
            "source": str(a.get("source") or "").strip().lower(),
        }
    return out


def _when(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _subject_of(doc_atoms: list[Any], info: dict[str, Any]) -> str:
    if info.get("subject"):
        return str(info["subject"])
    for a in doc_atoms:
        thread = _value(a).get("email_thread")
        if isinstance(thread, dict) and thread.get("subject"):
            return str(thread["subject"]).strip()
    return ""


def bundle_documents(
    by_doc: "OrderedDict[str, list[Any]]",
    index: dict[str, dict[str, Any]] | None = None,
) -> "OrderedDict[str, dict[str, Any]]":
    """Group documents into conversations. Every message under one thread
    subject is one bundle; a file joins the bundle of the message that carried
    it -- by the attachment id the manifest recorded, else by having been
    mirrored from the CRM within two minutes of exactly one such message.
    Everything else is a bundle of its own. Each bundle records how each
    document joined it ("thread", "attachment", "arrived_with", "alone")."""
    index = index or {}
    bundles: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    threads: dict[str, str] = {}
    pending: list[tuple[str, str, dict[str, Any]]] = []

    def _new(key: str, title: str) -> dict[str, Any]:
        b = {"title": title, "docs": [], "links": {}, "attachment_ids": set(), "carried_at": []}
        bundles[key] = b
        return b

    for key, doc_atoms in by_doc.items():
        filename = _doc_filename(doc_atoms[0], key)
        info = index.get(filename) or {}
        subject = _subject_of(doc_atoms, info)
        if subject:
            tk = "thread:" + thread_key(subject)
            b = bundles.get(threads.get(tk, "")) or _new(tk, subject)
            threads.setdefault(tk, tk)
            b["docs"].append(key)
            b["links"][key] = "thread"
            b["attachment_ids"].update(info.get("attachment_ids") or [])
            if info.get("attachment_ids"):
                b["carried_at"].append(_when(str(info.get("authored_at") or "")))
            continue
        pending.append((key, filename, info))

    for key, filename, info in pending:
        joined = None
        ext = str(info.get("external_id") or "")
        file_id = ext.split(":", 1)[1] if ":" in ext else ""
        if file_id:
            for bk, b in bundles.items():
                if bk.startswith("thread:") and file_id in b["attachment_ids"]:
                    joined, how = bk, "attachment"
                    break
        if joined is None and info.get("source") == "hubspot":
            at = _when(str(info.get("authored_at") or ""))
            near = [
                bk for bk, b in bundles.items()
                if bk.startswith("thread:") and at is not None
                and any(t is not None and abs(at - t) <= _ARRIVED_WITH for t in b["carried_at"])
            ]
            if len(near) == 1:
                joined, how = near[0], "arrived_with"
        if joined is None:
            b = _new(key, _title(by_doc[key], filename))
            b["docs"].append(key)
            b["links"][key] = "alone"
        else:
            bundles[joined]["docs"].append(key)
            bundles[joined]["links"][key] = how
    return bundles


def deal_name_from_manifest(project_dir: Path | str | None) -> str:
    """The CRM deal name from the manifest sidecar, or ''."""
    if not project_dir:
        return ""
    path = Path(project_dir) / ".parser_manifest.json"
    if not path.is_file():
        return ""
    try:
        ctx = json.loads(path.read_text(encoding="utf-8")).get("context") or {}
        crm = ctx.get("crm") if isinstance(ctx, dict) else None
        return str((crm or {}).get("deal_name") or "").strip()
    except (OSError, ValueError, AttributeError):
        return ""


def _doc_filename(atom: Any, fallback: str = "") -> str:
    """The file an atom came from. EvidenceAtom carries it on its source ref,
    not as an attribute -- ``source_filename`` exists only in the serialized
    atoms.json. Live 010162 run 7 (compile 752dbfbd, 2026-09-15): looked up by
    attribute, every manifest lookup missed, the kiosk packing list never
    joined the message it arrived with and was judged alone as this deal, and
    every non-email document was titled by its artifact id."""
    v = getattr(atom, "source_filename", None)
    if v:
        return str(v)
    try:
        for ref in getattr(atom, "source_refs", None) or []:
            fn = getattr(ref, "filename", None) or (ref.get("filename") if isinstance(ref, dict) else None)
            if fn:
                return str(fn)
    except Exception:
        pass
    val = getattr(atom, "value", None)
    if isinstance(val, dict) and val.get("source_filename"):
        return str(val["source_filename"])
    return fallback


def _atom_type(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return str(getattr(at, "value", at) or "")


def _value(atom: Any) -> dict:
    v = getattr(atom, "value", None)
    return v if isinstance(v, dict) else {}


def _title(atoms: list[Any], filename: str) -> str:
    """What the document calls itself: an email thread's subject, a note's title, else the filename."""
    for a in atoms:
        v = _value(a)
        thread = v.get("email_thread")
        if isinstance(thread, dict) and thread.get("subject"):
            return str(thread["subject"]).strip()
    for a in atoms:
        v = _value(a)
        if v.get("kind") in ("hubspot_note_meta", "hubspot_note_body") and v.get("title"):
            return str(v["title"]).strip()
    return filename


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def common_lines(atoms: list[Any]) -> set[str]:
    """Lines that recur across documents of one deal: banners ("External sender
    Check the sender..."), signatures, disclaimers. They describe no job, so the
    judge should not read them as the document's opening lines. No vocabulary:
    a line is chrome because the deal repeats it, whatever it says."""
    seen: dict[str, set[str]] = {}
    for a in atoms:
        # The opening of the line: a banner repeats its opening even when one
        # copy was joined to itself with a separator.
        key = _norm(getattr(a, "raw_text", ""))[:80]
        if len(key) < 12:
            continue
        doc = str(getattr(a, "source_artifact_id", None) or getattr(a, "artifact_id", None) or "")
        seen.setdefault(key, set()).add(doc)
    return {k for k, docs in seen.items() if len(docs) >= 2}


def _own_words_first(atoms: list[Any]) -> list[Any]:
    """An email's own message before the history it quotes, else document order."""
    own = [a for a in atoms if _value(a).get("quoted") is False]
    if not own:
        return list(atoms)
    rest = [a for a in atoms if _value(a).get("quoted") is not False]
    return own + rest


def _first_lines(atoms: list[Any], skip: set[str], limit: int) -> list[str]:
    lines: list[str] = []
    for a in _own_words_first(atoms):
        v = _value(a)
        if v.get("kind") in _META_KINDS or v.get("field_name") in _META_KINDS or v.get("non_deal") or _atom_type(a) in ("raw_utterance",):
            continue
        t = " ".join(str(getattr(a, "raw_text", "") or "").split())
        if not t or t in lines or _norm(t)[:80] in skip or t.startswith("[Image extracted"):
            continue
        lines.append(t[:_LINE_CHARS])
        if len(lines) >= limit:
            break
    if not lines:  # a transcript is nothing but utterances
        for a in atoms:
            t = " ".join(str(getattr(a, "raw_text", "") or "").split())
            if t and len(t) > 40:
                lines.append(t[:_LINE_CHARS])
            if len(lines) >= limit:
                break
    return lines


_OPENER_LINES = 4


def bundle_lines(docs: list[list[Any]], common: set[str] | None = None) -> list[str]:
    """The conversation's first lines: the opening document's first few, since
    a thread's opener carries the ask (000061: the customer's "we have
    identified a need for wireless access point height adjustments"), then one
    line from each document in turn so a long message does not crowd out the
    file that arrived with it. ``docs`` are in the order they were written."""
    skip = common or set()
    per_doc = [_first_lines(d, skip, _LINES) for d in docs]
    lines: list[str] = []
    if per_doc:
        for l in per_doc[0][:_OPENER_LINES]:
            if l not in lines:
                lines.append(l)
    i = 0
    while len(lines) < _LINES and any(i < len(p) for p in per_doc):
        for p in per_doc:
            if i < len(p) and p[i] not in lines:
                lines.append(p[i])
                if len(lines) >= _LINES:
                    break
        i += 1
    return lines


def bundle_text(title: str, docs: list[list[Any]], common: set[str] | None = None) -> str:
    """The conversation as the judge reads it: its subject and its first lines."""
    return f"DOCUMENT: {title}\n" + "\n".join(f"- {l}" for l in bundle_lines(docs, common))


def document_text(atoms: list[Any], filename: str, common: set[str] | None = None) -> str:
    """One document as the judge reads it: its title and its first lines."""
    return bundle_text(_title(atoms, filename), [atoms], common)


def judge_documents(
    atoms: list[Any],
    *,
    deal_name: str,
    project_id: str = "",
    project_dir: Path | str | None = None,
    index: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    """Partition ``atoms`` into (kept, dropped) and describe each verdict.

    Every conversation is judged once and the verdict covers every document
    in it. A confident ``other_job`` drops their atoms; anything else keeps
    them. Without a deal name there is nothing to compare against, so nothing
    is judged.
    """
    if not deal_name or not atoms:
        return list(atoms), [], []
    try:
        from app.core.decide import DecisionScope, decide
    except Exception:  # pragma: no cover
        return list(atoms), [], []

    by_doc: "OrderedDict[str, list[Any]]" = OrderedDict()
    for a in atoms:
        key = str(getattr(a, "source_artifact_id", None) or getattr(a, "artifact_id", None) or "")
        by_doc.setdefault(key, []).append(a)

    scope = DecisionScope(deal_id=str(project_id or ""))
    dropped_ids: set[int] = set()
    verdicts: list[dict[str, Any]] = []
    common = common_lines(atoms)
    index = index if index is not None else manifest_index(project_dir)
    bundles = bundle_documents(by_doc, index)
    model, timeout, floor = judge_model(), judge_timeout(), min_confidence()
    idx = index if index is not None else manifest_index(project_dir)
    for b in bundles.values():
        keys = list(b["docs"])
        # In the order they were written, so the opener's lines lead.
        keys.sort(key=lambda k: _when(str((idx.get(_doc_filename(by_doc[k][0], k)) or {}).get("authored_at") or "")) or datetime.max.replace(tzinfo=timezone.utc))
        docs = [by_doc[k] for k in keys]
        filenames = [_doc_filename(d[0], k) for d, k in zip(docs, keys)]
        # The model reads TEXT (first 600 chars) and Context (first 1200): the
        # deal and the conversation's subject go in the text -- that is what a
        # taught verdict is matched on -- and the conversation's lines go in
        # the context. Live 000061 (compile 3f79093a): with everything in the
        # text, the judge saw the subject and four greetings and set the
        # customer's own quote-consultation thread aside as another job.
        # The store compares the TEXT alone, so the text must be the thing a
        # lesson is about: the document. With the deal name in front, every
        # bundle of one deal shared most of its embedding, and a single taught
        # `other_job` matched all of them (dev 2026-09-16 02:47Z, 010162: one
        # lesson for the Delta close-down thread emptied the whole deal to 12
        # atoms). The DEAL line the model reads moves into the context; the
        # subject loses its reply markers so one lesson covers the thread.
        subject = re.sub(_SUBJECT_PREFIX_RE.pattern, "", str(b["title"]).strip(), flags=re.I).strip() or str(b["title"]).strip()
        text = f"DOCUMENT: {subject}"
        context = f"DEAL: {deal_name.strip()}\n" + "\n".join(f"- {l}" for l in bundle_lines(docs, common))
        try:
            # Judgments only: a person's or a Deal Kit's verdict decides; the
            # model's own cached verdicts never do, since one wrong other_job
            # would then remove a whole thread on every later compile.
            d = decide(RELATION, text[:600], CANDIDATES, instruction=INSTRUCTION, context=context[:1200],
                       scope=scope, model=model, timeout=timeout, exclude_created_by=("teacher",))
        except Exception:
            d = None
        verdict = getattr(d, "verdict", None) or "this_deal"
        conf = float(getattr(d, "confidence", 0.0) or 0.0)
        source = getattr(d, "source", "fallback")
        other = verdict == "other_job" and (source == "store" or conf >= floor)
        n_atoms = sum(len(d_) for d_ in docs)
        verdicts.append({
            "bundle": b["title"], "filename": filenames[0], "filenames": filenames,
            "links": {fn: b["links"][k] for fn, k in zip(filenames, b["docs"])},
            "verdict": "other_job" if other else "this_deal", "model_verdict": verdict,
            "confidence": round(conf, 3), "source": source, "atoms": n_atoms,
            "correction_id": getattr(d, "correction_id", None),
        })
        if other:
            for doc_atoms in docs:
                dropped_ids.update(id(a) for a in doc_atoms)
                for a in doc_atoms:
                    flags = list(getattr(a, "review_flags", None) or [])
                    if "other_job" not in flags:
                        flags.append("other_job")
                    try:
                        a.review_flags = flags
                    except Exception:
                        pass
    kept = [a for a in atoms if id(a) not in dropped_ids]
    dropped = [a for a in atoms if id(a) in dropped_ids]
    return kept, dropped, verdicts


def verdict_note(v: dict[str, Any]) -> str:
    """One trace line per conversation, kept or set aside."""
    what = f"{v['bundle'][:80]} ({v['atoms']} atoms, {len(v.get('filenames') or [v['filename']])} document(s))"
    how = f"{v['model_verdict'] or 'undecided'} {v['confidence']:.2f} {v['source']}"
    if v["verdict"] == "other_job":
        return f"INFO: document_job_scope set aside {what}; {how}"
    return f"INFO: document_job_scope kept {what}; {how}"


__all__ = ["RELATION", "CANDIDATES", "INSTRUCTION", "enabled", "deal_name_from_manifest", "common_lines",
           "thread_key", "manifest_index", "bundle_documents", "bundle_lines", "bundle_text", "document_text",
           "judge_documents", "verdict_note", "judge_model", "judge_timeout", "min_confidence"]
