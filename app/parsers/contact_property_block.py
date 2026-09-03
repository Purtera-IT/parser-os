"""Two people, one property block: read a site's contacts the way sites are read.

    On Site Contact Name | Rosalyn Hemingway | OSC email | Rosalyn.Hemingway@Sodexo.com
    OSC phone             | 843-423-8335 x3069 | Backup Contact | Bernard Donnelly
    Backup email          | bernie.donnelly@sodexo.com | Backup Phone | 404-918-0783

Nothing reads this. Downstream, a generic heuristic re-parses the flattened row
TEXT and assigns fields by POSITION -- cell 0 is "name", cell 1 is "email" --
without looking at what the cell actually says. On row 2 that gives
name="OSC phone", email="843-423-8335 x3069": a label read as a person, a phone
read as an address. On row 1 it happens to come out right, because the real
name and email land in the positions the heuristic assumes. Measured on the
ten 010215 SOWs: every school's on-site phone is present in its own document
and absent from every downstream atom.

The same doctrine as site_property_block: SHAPE decides what a value IS -- an
email matches EMAIL, a phone matches PHONE, a name matches neither and reads as
a person -- so a vendor who writes "POC" instead of "OSC" or reorders the rows
still produces a correct contact. Only WHICH person a value belongs to (the
on-site contact vs. their backup) is a label call, because shape cannot tell
two people apart. Get that label wrong and the value still lands on a real
person, never in the void: a field with no role match is not dropped, it is
kept as an unassigned contact rather than mis-attributed to the wrong one.
"""

from __future__ import annotations

import re
from typing import Any

from app.parsers.value_shapes import classify_value

_ROLE_LABELS: dict[str, tuple[str, ...]] = {
    "primary_name": ("on site contact name", "onsite contact name", "osc name",
                      "on-site contact name", "point of contact", "poc name",
                      "primary contact", "site contact"),
    "primary_email": ("osc email", "on site contact email", "onsite email",
                       "poc email", "primary email", "site contact email"),
    "primary_phone": ("osc phone", "on site contact phone", "onsite phone",
                       "poc phone", "primary phone", "site contact phone"),
    "backup_name": ("backup contact", "backup contact name", "alternate contact",
                     "secondary contact", "escalation contact"),
    "backup_email": ("backup email", "alternate email", "secondary email",
                      "escalation email"),
    "backup_phone": ("backup phone", "alternate phone", "secondary phone",
                      "escalation phone"),
}

_NAME_SHAPE = re.compile(r"^[A-Z][a-zA-Z'.-]+(?:\s+[A-Z][a-zA-Z'.-]+){1,3}$")
_LABEL_LIKE = re.compile(r"[:?#]\s*$")
_NOISE = re.compile(r"^(n/?a|none|tbd|-|—|)$", re.I)


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _match_role_label(text: str) -> str | None:
    t = _norm(text).rstrip(":").lower()
    for field, names in _ROLE_LABELS.items():
        if t in names:
            return field
    return None


def _shape_ok(field: str, value: str) -> bool:
    """The value must actually BE what the label claims, or it is not taken.

    This is what stops "OSC phone" (a label, not a name) landing in a name
    field, and a phone number landing in an email field, when the row's real
    shape does not match the position a naive reader assumed.
    """
    kind = classify_value(value)
    if field.endswith("_email"):
        return kind == "email"
    if field.endswith("_phone"):
        return kind == "phone"
    if field.endswith("_name"):
        return kind is None and bool(_NAME_SHAPE.match(value))
    return False


def fields_from_contact_row(cells: dict | None) -> dict[str, str]:
    """Read role-tagged contact fields out of one property-block row.

    Mirrors ``site_property_block.fields_from_property_row``: cells arrive as
    label, value, label, value ... in order; the dict keys are useless (the
    merged section header, repeated) so pairing is positional. A label is only
    honoured when the very next cell both exists and passes the shape check for
    that field -- a value is never taken on label match alone.
    """
    if not isinstance(cells, dict) or not cells:
        return {}
    values = [_norm(v) for v in cells.values()]
    out: dict[str, str] = {}
    for i, cell in enumerate(values):
        field = _match_role_label(cell)
        if not field:
            continue
        value = values[i + 1] if i + 1 < len(values) else ""
        if not value or _NOISE.match(value) or _match_role_label(value):
            continue
        if not _shape_ok(field, value):
            continue
        out.setdefault(field, value)
    return out


def people_from_multiline_cells(cells: dict | None) -> list[dict[str, Any]]:
    """A person typed into ONE cell across lines, under any label.

        TECH Contact Information (Main) | "Bernie Donnelly\n404-918-0783" | Escalation Contact | ""

    Neither the role-label reader nor the LLM handled this: the live record
    came out with the phone in the email field. Split the cell on newlines and
    let each line type itself -- a name-shaped line is the name, an email line
    the email, a phone line the phone. The label cell becomes the role verbatim.
    """
    if not isinstance(cells, dict) or not cells:
        return []
    values = list(cells.values())
    out: list[dict[str, Any]] = []
    for i in range(1, len(values)):
        label, raw = _norm(values[i - 1]), str(values[i] or "")
        if not label or "\n" not in raw or _match_role_label(label):
            continue
        lines = [_norm(x) for x in raw.split("\n") if _norm(x)]
        if len(lines) < 2:
            continue
        person: dict[str, Any] = {"role": _LABEL_LIKE.sub("", label).strip(), "kind": "person"}
        for ln in lines:
            kind = classify_value(ln)
            if kind == "email" and "email" not in person:
                person["email"] = ln
            elif kind == "phone" and "phone" not in person:
                person["phone"] = ln
            elif kind is None and _NAME_SHAPE.match(ln) and "name" not in person:
                person["name"] = ln
        if person.get("name") and (person.get("email") or person.get("phone")):
            out.append(person)
    return out


def contacts_from_property_rows(rows: list[dict | None]) -> list[dict[str, Any]]:
    """Merge a document's contact rows into up to two people: primary + backup.

    Requires a name, an email, or a phone for a person to be emitted -- a role
    label with nothing shape-valid behind it names no one.
    """
    merged: dict[str, str] = {}
    for cells in rows or []:
        for k, v in fields_from_contact_row(cells).items():
            merged.setdefault(k, v)

    people: list[dict[str, str]] = []
    for prefix, role in (("primary", "On Site Contact"), ("backup", "Backup Contact")):
        name = merged.get(f"{prefix}_name")
        email = merged.get(f"{prefix}_email")
        phone = merged.get(f"{prefix}_phone")
        if not (name or email or phone):
            continue
        person: dict[str, str] = {"role": role, "kind": "person"}
        if name:
            person["name"] = name
        if email:
            person["email"] = email
        if phone:
            person["phone"] = phone
        people.append(person)
    seen_names = {p.get("name") for p in people}
    for cells in rows or []:
        for p in people_from_multiline_cells(cells):
            if p.get("name") not in seen_names:
                people.append(p)
                seen_names.add(p.get("name"))
    return people
