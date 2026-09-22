"""What did the parser NOT read?

A recall miss leaves nothing behind: a paragraph that produced no atom makes
no card, no flag, no row. The only way anyone found out was a PM reading the
source next to the output and saying "out of that whole paragraph you have one
atom" -- which is not a process, it is luck.

So every text artifact is diffed against its own atoms: each line of the
source is CLAIMED (some atom quotes it), SUPPRESSED (an atom existed and a
gate dropped it, reason attached) or UNREAD (nothing ever came of it). The
unclaimed lines ride in the envelope, the labeler shows them greyed under the
message they belong to, and a PM promotes one in a click instead of hunting.

Deliberately blind to nothing: mail headers, quote markers and signature
chrome are listed too, marked ``chrome``, so the PM can see the parser's
judgement rather than trust it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

#: Text we can read back and diff. A PDF's line breaks are a rendering, not
#: the source, so it is out of scope here.
TEXT_SUFFIXES = {".eml", ".txt", ".md", ".msg", ".html", ".htm"}

_CHROME_RE = re.compile(
    r"^\s*(?:"
    r"(?:from|to|cc|bcc|sent|date|subject|importance|reply-to|message-id)\s*:|"
    r">+|_{5,}|-{5,}|={5,}|\*{3,}|"
    r"(?:cell|mobile|office|direct|tel|phone|p|o|m|f|fax|email|e)\s*:\s*\S|"
    r"\[cid:|\[Image|<https?://|https?://\S+$|"
    r"(?:thanks|thank you|regards|best|sincerely|cheers)[,!.]?\s*$|"
    r"(?:the )?contents of this (?:e-?mail|message) are intended|"
    r"this (?:e-?mail|message) (?:and any attachments |)(?:is|are|may be) (?:confidential|privileged)|"
    r"if you (?:are not|have received) (?:the|this)|"
    r"team inbox\s*:|"
    r"hubspot note(?: id)?\s*:|note_id=|"
    # a line that is ONLY an address, a domain, a wrapped link or a phone
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:<[^>]*>)?\s*$|"
    r"[A-Za-z0-9.-]+\.(?:com|net|org|io|co)(?:<[^>]*>)?\s*$|"
    r"[+(]?\d[\d\s().-]{8,}\s*$"
    r")",
    re.I,
)

#: Outlook writes an inline image as "[A screenshot of a phone   AI-generated
#: content may be incorrect., Picture, Picture]". The picture is real content
#: nobody read -- 010289's door diagram arrived exactly this way.
_IMAGE_PLACEHOLDER_RE = re.compile(r"^\s*\[[^\]]*(?:picture|image|screenshot|logo|cid:)[^\]]*\]?\s*$", re.I)

#: A line shorter than this cannot carry a fact on its own.
_MIN_CHARS = 12


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except Exception:
        return ""
    if path.suffix.lower() in {".eml", ".msg"}:
        try:
            import email
            from email import policy

            msg = email.message_from_bytes(raw, policy=policy.default)
            body = msg.get_body(preferencelist=("plain", "html"))
            if body is not None:
                return body.get_content()
        except Exception:
            pass
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return ""


def _atom_texts(atoms: list[Any], artifact_id: str | None) -> list[str]:
    """Every string an atom quotes. ``artifact_id=None`` means every atom of
    the deal, whichever document it came from."""
    out = []
    for a in atoms:
        if artifact_id is not None and str(getattr(a, "artifact_id", "") or "") != artifact_id:
            continue
        for t in (getattr(a, "raw_text", ""), getattr(a, "normalized_text", "")):
            n = _norm(t)
            if n:
                out.append(n)
        v = getattr(a, "value", None)
        if isinstance(v, dict):
            for k in ("quote", "text", "value"):
                n = _norm(v.get(k) or "")
                if n:
                    out.append(n)
            # A label consumed into its items ("Provided by us:") was read --
            # it lives on every item as the lead-in.
            for k in ("list_label", "parent_field", "field_name"):
                n = _norm(v.get(k) or "")
                if n:
                    out.append(n)
            for lead in v.get("lead_in") or []:
                n = _norm(lead)
                if n:
                    out.append(n)
        loc = getattr(a, "locator", None)
        if isinstance(loc, dict):
            for lead in loc.get("lead_in") or []:
                n = _norm(lead)
                if n:
                    out.append(n)
    return out


def coverage_for_artifact(
    path: Path,
    artifact_id: str,
    kept: list[Any],
    suppressed: list[Any] | None = None,
    claimed_anywhere: list[str] | None = None,
) -> dict[str, Any]:
    """Line-by-line: what became an atom, what was dropped, what was never read."""
    text = _read_text(path)
    if not text:
        return {}
    # A signature, a quoted history, a repeated ask: read ONCE for the deal on
    # purpose. Looking only at this artifact's atoms, every later copy reads as
    # a miss -- live 010289 reported 84, nearly all of them "Account
    # Executive" and a phone number in mail number seven.
    claimed = _atom_texts(kept, artifact_id) + list(claimed_anywhere or [])
    dropped = _atom_texts(list(suppressed or []), artifact_id)
    lines: list[dict[str, Any]] = []
    n_claimed = 0
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        n = _norm(line)
        if not n:
            continue
        if any(n in c or c in n for c in claimed):
            n_claimed += 1
            continue
        if _IMAGE_PLACEHOLDER_RE.match(line):
            # A picture sat in this mail and nothing read it. Not chrome, not
            # prose we missed: its own answer ("go look at the image").
            state = "image"
        elif _CHROME_RE.match(line) or len(line) < _MIN_CHARS:
            state = "chrome"
        else:
            state = "unread"
        if any(n in d or d in n for d in dropped):
            state = "suppressed"
        lines.append({"line": i, "text": line[:400], "state": state})
    total = n_claimed + len(lines)
    return {
        "artifact_id": artifact_id,
        "lines_total": total,
        "lines_claimed": n_claimed,
        "unclaimed": lines[:200],
        "unread_count": sum(1 for x in lines if x["state"] == "unread"),
        "image_count": sum(1 for x in lines if x["state"] == "image"),
    }


def build_text_coverage(
    artifact_paths: dict[str, Path],
    atoms: list[Any],
    suppressed: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """One row per readable text artifact. Never raises: coverage reporting
    must not be able to fail a compile."""
    out: list[dict[str, Any]] = []
    # Long enough to be a fact rather than a coincidence, and to keep this
    # deal-wide pass from swallowing a real miss.
    everywhere = [t for t in _atom_texts(atoms, None) if len(t) >= 18]
    for artifact_id, path in (artifact_paths or {}).items():
        try:
            p = Path(path)
            if p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            row = coverage_for_artifact(p, str(artifact_id), atoms, suppressed, everywhere)
            if row:
                out.append(row)
        except Exception:
            continue
    return out


__all__ = ["build_text_coverage", "coverage_for_artifact", "TEXT_SUFFIXES"]
