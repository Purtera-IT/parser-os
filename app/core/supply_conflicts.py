"""Two documents, one part, two different companies buying it.

Deal 010288 arrived with a reseller's supply list and the vendor drawing it
was quoting from. Both say who provides each piece and they do not agree: four
parts the email puts under "Provided by us" are printed on the drawing as
"Installer supplied Components". The email says so itself -- "the 'Installer
Supplied Components' are not accurate, as we provide several of those pieces"
-- and never says which several. Finding them took reading eighteen labels off
a picture and laying them against a ten-line list by hand.

Both sides are now atoms with a heading, so the comparison is mechanical.

THE TRAP IS FLAGGING AGREEMENTS. Eight parts appear on both documents, and a
rule that fires whenever the two headings differ in wording flags all eight --
including "Provided by Club/installer" against "Installer supplied
Components", which is the same answer said twice. What matters is not the
words but WHICH SIDE of the document's own authorship a heading names:

  * "Provided by us" on a CDW email names CDW.
  * "Huzzard supplied Components" on a Huzzard drawing names Huzzard.

Both are that document saying "we do". A heading naming anyone else is that
document saying "somebody else does". A conflict is one document claiming a
part for itself while another hands the same part to a third party -- which
leaves "Provided by us" against "Huzzard supplied" correctly silent, because a
reseller shipping its vendor's kit is not a disagreement, it is a supply
chain.
"""
from __future__ import annotations

import re
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)

VERSION = "supply_conflicts_v1"

#: A heading that assigns a part to somebody. Deliberately narrow: it has to
#: be about supply, or every bulleted section in every email becomes a claim.
_SUPPLY_HEADING_RE = re.compile(
    r"\b(provided|supplied|furnished|sourced|by others|responsib)", re.I)
#: The document speaking for itself.
_FIRST_PERSON_RE = re.compile(r"\b(us|we|our|ours|ourselves)\b", re.I)
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")
#: Short names collide: "Relay" is a part, "PC" is half of twenty things.
_MIN_ITEM_CHARS = 4


def _norm(text: str) -> str:
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", str(text or "").lower())).strip()


def _locator(atom: Any) -> dict:
    refs = getattr(atom, "source_refs", None) or []
    loc = getattr(refs[0], "locator", None) if refs else None
    return loc if isinstance(loc, dict) else {}


def _surface(atom: Any) -> str:
    """Which document a line came off.

    Not the artifact: a drawing linked from an email is READ ONTO that email,
    so the vendor's parts list and the reseller's supply list share an
    artifact id while being two entirely separate statements of who buys what.
    The sheet is what separates them -- an atom read off BPW061725 Rev1 is not
    a line the sender of the mail typed.
    """
    loc = _locator(atom)
    return f"{getattr(atom, 'artifact_id', '') or ''}#{loc.get('sheet') or ''}"


def _heading(atom: Any) -> str:
    """The supply heading this line sits under, if it sits under one."""
    loc = _locator(atom)
    for key in ("section_path", "lead_in"):
        got = loc.get(key)
        if isinstance(got, list) and got:
            head = str(got[0] or "").strip().rstrip(":")
            if head and _SUPPLY_HEADING_RE.search(head):
                return head
    return ""


def _authors(atom: Any, documents: dict[str, dict]) -> list[str]:
    """Names this document speaks as: its sender's org, and its own vendor."""
    out: list[str] = []
    loc = _locator(atom)
    doc = documents.get(str(getattr(atom, "artifact_id", "") or "")) or {}
    for got in (loc.get("sender"), doc.get("sender_email"), doc.get("sender_domain")):
        name = str(got or "")
        if "@" in name:
            name = name.split("@", 1)[1]
        name = name.split(".")[0].strip()
        if len(name) >= 3:
            out.append(name.lower())
    # A drawing speaks as whoever drew it; the sheet id carries their name.
    value = getattr(atom, "value", None) or {}
    for got in (value.get("vendor"), loc.get("sheet")):
        name = _norm(got).split(" ")[0] if got else ""
        if len(name) >= 3:
            out.append(name)
    return out


def side_of(heading: str, authors: list[str]) -> str:
    """``"self"`` when the heading names the document's own side, else ``"other"``.

    A reseller writing "Provided by us" and a vendor writing "Huzzard supplied"
    on Huzzard's own drawing are both saying the same kind of thing.
    """
    low = _norm(heading)
    if _FIRST_PERSON_RE.search(heading):
        return "self"
    if any(a and a in low for a in authors):
        return "self"
    return "other"


def _item_key(text: str) -> str:
    return _norm(text)


def _same_item(a: str, b: str) -> bool:
    """One part named twice. A drawing prints the short name and a quote spells
    it out: "Power Supply" against "Power Supply for mag lock/locking
    mechanism"."""
    if not a or not b or len(a) < _MIN_ITEM_CHARS or len(b) < _MIN_ITEM_CHARS:
        return False
    return a == b or a in b or b in a


def find_supply_conflicts(atoms: list[Any], documents: dict[str, dict] | None = None,
                          ) -> list[EvidenceAtom]:
    """One ``open_question`` per part two documents hand to different companies."""
    docs = documents or {}
    claims: list[tuple[Any, str, str, str, str]] = []
    for atom in atoms or []:
        try:
            head = _heading(atom)
            if not head:
                continue
            key = _item_key(getattr(atom, "raw_text", "") or "")
            if len(key) < _MIN_ITEM_CHARS:
                continue
            claims.append((atom, key, head, side_of(head, _authors(atom, docs)),
                           _surface(atom)))
        except Exception:
            continue

    # Group by the two headings that disagree, not by part. Four cards saying
    # the same sentence about different nouns is both unreadable and unstable:
    # they are 0.95 similar, and near-duplicate collapse eats them.
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    seen: set[str] = set()
    for i, (a1, k1, h1, s1, f1) in enumerate(claims):
        for a2, k2, h2, s2, f2 in claims[i + 1:]:
            # One document disagreeing with itself is a drafting problem, not
            # a supply question.
            if f1 == f2:
                continue
            # Both documents claiming the part for their own side is a supply
            # chain (a reseller shipping its vendor's kit), not a conflict.
            # Both handing it to a third party is agreement.
            if s1 == s2 or not _same_item(k1, k2):
                continue
            # ONE ENTRY PER PART, not per pair of lines. A quote line can name
            # two things -- "USB Cable connecting PC to RS232 to USB converter"
            # matches both the drawing's "USB Cable" and its "RS232 to USB
            # converter" -- and a part is in dispute once.
            ident = k1 if len(k1) <= len(k2) else k2
            if ident in seen:
                continue
            seen.add(ident)

            mine, theirs = (a1, h1) if s1 == "self" else (a2, h2)
            other = (a2, h2) if s1 == "self" else (a1, h1)
            key = (theirs, other[1])
            slot = pairs.setdefault(key, {"anchor": mine, "items": []})
            named = min(((getattr(mine, "raw_text", "") or "").strip(),
                         (getattr(other[0], "raw_text", "") or "").strip()), key=len)
            slot["items"].append({
                "item": named,
                "claimed_text": (getattr(mine, "raw_text", "") or "").strip(),
                "assigned_text": (getattr(other[0], "raw_text", "") or "").strip(),
                "atom_ids": [str(getattr(mine, "id", "")), str(getattr(other[0], "id", ""))],
            })

    return [_ask(head_self, head_other, slot) for (head_self, head_other), slot in pairs.items()]


def _ask(head_self: str, head_other: str, slot: dict[str, Any]) -> EvidenceAtom:
    """The question a PM answers once, naming every part in dispute."""
    anchor = slot["anchor"]
    items = slot["items"]
    names = [x["item"] for x in items]
    listed = ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]
    count = f"{len(names)} part" + ("s are" if len(names) > 1 else " is")
    text = (f"{count} claimed by both sides: {listed}. One document puts them under "
            f"\"{head_self}\" and another under \"{head_other}\". Who supplies them?")

    artifact_id = str(getattr(anchor, "artifact_id", "") or "")
    atom_id = stable_id("atm", artifact_id, VERSION, head_self, head_other,
                        "|".join(sorted(names)))
    refs = getattr(anchor, "source_refs", None) or []
    filename = (getattr(refs[0], "filename", "") if refs else "") or ""
    src = SourceRef(
        id=stable_id("src", atom_id),
        artifact_id=artifact_id,
        # The question belongs to the document that claimed the parts; when
        # that source carries no type, an email is the honest default -- a
        # supply list is something somebody wrote to somebody.
        artifact_type=(getattr(refs[0], "artifact_type", None) if refs else None)
        or ArtifactType.email,
        filename=filename,
        locator={"extraction": VERSION, "headings": [head_self, head_other]},
        extraction_method=VERSION,
        parser_version=VERSION,
    )
    return EvidenceAtom(
        id=atom_id,
        project_id=str(getattr(anchor, "project_id", "") or ""),
        artifact_id=artifact_id,
        atom_type=AtomType.open_question,
        raw_text=text,
        normalized_text=text,
        # Declares itself generated so open-question resolution does not
        # "answer" it with the very BOM lines it was built from.
        value={"kind": "supply_conflict",
               "via": VERSION, "claimed_by_sender": head_self,
               "assigned_elsewhere": head_other, "items": items,
               "atom_ids": [i for x in items for i in x["atom_ids"]]},
        entity_keys=[],
        source_refs=[src],
        receipts=[],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.62,
        confidence_raw=0.62,
        calibrated_confidence=0.62,
        review_status=ReviewStatus.needs_review,
        review_flags=[VERSION],
        parser_version=VERSION,
    )


__all__ = ["find_supply_conflicts", "side_of", "VERSION"]
