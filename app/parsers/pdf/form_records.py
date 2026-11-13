"""One form record, one atom: a PDF form's address read as one line.

A freight or site questionnaire prints one address as separate fields:

    SITE ADDRESS (Pickup Location): 615 N 48th St
    CITY: Phoenix
    STATE: AZ
    ZIP: 85008

The page parse and the form-answer pass each emit one atom per field, so the
address reached the labeling page as four lines, and three of them ("CITY:
Phoenix", "STATE: AZ", "ZIP: 85008") meant nothing on their own. Labelers
were dropping them as fragments by hand on every form, and the site head saw
a street with no city. Live 000037 (Maricopa County freight form).

This pass reads what the form already says. It never infers: a piece joins
an address only when its own label names it a city / state / ZIP / country /
second address line, its value has that shape, it sits on the same page, and
the address it joins does not already have that field. Everything the pieces
held stays on the joined atom (``value["form_record"]``), and the texts the
atoms had before are listed in ``value["joined_from"]`` so a saved label can
be moved onto the joined line (``scripts/move_joined_form_labels.py``).

Two smaller shapes ride along:

* The form-answer pass is additive: it re-emits a field the page text already
  carries ("SITE CONTACT (for Pickup): Adam ..." twice). The copy is dropped
  and the survivor lists it in ``joined_from`` as a duplicate.
* Nothing else changes. Phone numbers stay their own line unless the page has
  exactly one contact field, because a form with an alternate contact makes
  "which contact is this phone" a guess, and the parser does not guess.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.normalizers import normalize_text

#: Pieces of an address and what their label may say.
_PIECE_LABELS: dict[str, tuple[str, ...]] = {
    "address2": ("address line 2", "address 2", "suite", "unit", "suite/unit", "apt"),
    "city": ("city", "town"),
    "state": ("state", "province", "st"),
    "zip": ("zip", "zip code", "postal code", "postcode", "zip/postal code"),
    "country": ("country",),
    "city_state_zip": ("city, state, zip", "city/state/zip", "city, state zip", "city state zip"),
}
#: Labels that open an address (the line the pieces join).
_ADDRESS_LABELS = (
    "address", "site address", "street address", "address line 1", "address 1", "street",
    "pickup address", "pick up address", "delivery address", "ship to address", "shipping address",
    "location address", "physical address", "service address",
)
_CONTACT_LABELS = ("site contact", "contact", "contact name", "poc", "point of contact", "onsite contact",
                   "on-site contact", "site poc")
_PHONE_LABELS = ("phone", "phone number", "phone #", "cell", "mobile", "telephone", "tel")
_EMAIL_LABELS = ("email", "e-mail", "email address")

#: "LABEL (aside): value" -- the label is short and ends at the first colon.
_FIELD_RE = re.compile(r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 /&#.,'()-]{0,60}?)\s*:\s*(?P<value>.*?)\s*$", re.S)
_PAREN_RE = re.compile(r"\([^)]*\)")
_NOISE = re.compile(r"^(n/?a|none|tbd|-|—|_+|☐.*|□.*)$", re.I)
_PIECE_ORDER = ("address2", "city", "state", "zip", "country")
_WINDOW = 12  # atoms: a piece further than this from its address is another record


def _page(a: Any) -> Any:
    try:
        return (a.source_refs[0].locator or {}).get("page") if a.source_refs else None
    except Exception:
        return None


def _is_form_answer(a: Any) -> bool:
    v = a.value if isinstance(a.value, dict) else {}
    return v.get("kind") == "form_answer"


def _label_key(label: str) -> str:
    t = _PAREN_RE.sub(" ", label or "")
    return re.sub(r"\s+", " ", t).strip(" :.-").lower()


def _field(text: str) -> tuple[str, str, str] | None:
    """``(label as printed, label key, value)`` when the whole text is ONE
    "Label: value" field, else None. A row of several fields ("CITY: Phoenix |
    PHONE: ...") is not one field and is left alone."""
    t = str(text or "").strip()
    if not t or "|" in t or "\n" in t:
        return None
    m = _FIELD_RE.match(t)
    if not m:
        return None
    label = m.group("label").strip()
    if len(_label_key(label).split()) > 5:
        return None
    return label, _label_key(label), m.group("value").strip()


def _piece_kind(key: str) -> str | None:
    for kind, names in _PIECE_LABELS.items():
        if key in names:
            return kind
    return None


def _is_address_label(key: str) -> bool:
    return key in _ADDRESS_LABELS or key.endswith(" address")


def _value_fits(kind: str, value: str) -> bool:
    from app.parsers.value_shapes import classify_value

    v = (value or "").strip()
    if not v or _NOISE.match(v) or len(v) > 60:
        return False
    shape = classify_value(v)
    if kind == "zip":
        return shape == "postal"
    if kind == "state":
        return shape == "state" or bool(re.fullmatch(r"[A-Za-z][A-Za-z .]{1,24}", v))
    if kind in ("city", "country"):
        return shape is None and bool(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,40}", v))
    if kind == "address2":
        return shape not in ("email", "phone")
    if kind == "city_state_zip":
        return bool(re.search(r"[A-Za-z]", v)) and shape not in ("email", "phone")
    return False


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower()).rstrip(" .;:")


def _note_joined(value: dict[str, Any], text: str, role: str) -> None:
    rows = list(value.get("joined_from") or [])
    if not any(r.get("text") == text and r.get("role") == role for r in rows):
        rows.append({"text": text, "role": role})
    value["joined_from"] = rows


def drop_duplicate_form_answers(atoms: list[Any]) -> list[Any]:
    """Drop a form-answer atom whose field the page text already carries.

    Same page; the page-text atom's text equals the answer's, or one of its
    " | " fields does. The survivor lists the copy in ``joined_from``."""
    by_page: dict[Any, list[tuple[int, set[str]]]] = {}
    for i, a in enumerate(atoms):
        if _is_form_answer(a):
            continue
        t = str(a.raw_text or "")
        segs = {_norm(t)} | {_norm(s) for s in t.split("|") if s.strip()}
        by_page.setdefault(_page(a), []).append((i, segs))
    drop: set[int] = set()
    survivors: dict[int, list[str]] = {}
    for i, a in enumerate(atoms):
        if not _is_form_answer(a):
            continue
        n = _norm(a.raw_text)
        if not n:
            continue
        for j, segs in by_page.get(_page(a), []):
            if n in segs:
                drop.add(i)
                survivors.setdefault(j, []).append(str(a.raw_text or ""))
                break
    if not drop:
        return atoms
    out: list[Any] = []
    for i, a in enumerate(atoms):
        if i in drop:
            continue
        if i in survivors:
            v = dict(a.value) if isinstance(a.value, dict) else {}
            for t in survivors[i]:
                _note_joined(v, t, "duplicate")
            a = a.model_copy(update={"value": v})
        out.append(a)
    return out


def join_address_pieces(atoms: list[Any]) -> list[Any]:
    """Fold a form's CITY / STATE / ZIP / country / second-line fields into the
    address field they belong to, on the same page, as one atom."""
    fields = [_field(a.raw_text) for a in atoms]
    heads = [i for i, f in enumerate(fields)
             if f and _is_address_label(f[1]) and f[2] and not _NOISE.match(f[2])]
    if not heads:
        return atoms
    taken: dict[int, dict[str, int]] = {h: {} for h in heads}
    absorbed: set[int] = set()
    for i, f in enumerate(fields):
        if not f or i in heads:
            continue
        kind = _piece_kind(f[1])
        if not kind or not _value_fits(kind, f[2]):
            continue
        slots = ("city", "state", "zip") if kind == "city_state_zip" else (kind,)
        page = _page(atoms[i])
        # The nearest address on the same page that still lacks this field:
        # the one above first, else the one below.
        cands = [h for h in heads if _page(atoms[h]) == page and abs(h - i) <= _WINDOW
                 and not any(s in taken[h] for s in slots)]
        if not cands:
            continue
        above = [h for h in cands if h < i]
        h = max(above) if above else min(cands)
        # Another address label between them means the piece is that one's.
        lo, hi = sorted((h, i))
        if any(lo < k < hi for k in heads if _page(atoms[k]) == page):
            continue
        for s in slots:
            taken[h][s] = i
        absorbed.add(i)
    if not absorbed:
        return atoms
    out: list[Any] = []
    for i, a in enumerate(atoms):
        if i in absorbed:
            continue
        if i in taken and taken[i]:
            label, _key, street = fields[i]
            pieces = sorted({p for p in taken[i].values()}, key=lambda p: min(
                _PIECE_ORDER.index(s) if s in _PIECE_ORDER else 9
                for s, q in taken[i].items() if q == p))
            record: dict[str, str] = {"address": street}
            parts_after_street: list[str] = []
            for p in pieces:
                _pl, pkey, pval = fields[p]
                kind = _piece_kind(pkey) or ""
                record[kind] = pval
                parts_after_street.append((kind, pval))
            # "615 N 48th St, Suite 2, Phoenix, AZ 85008, USA"
            seg: list[str] = []
            for kind, val in parts_after_street:
                if kind == "zip" and seg and record.get("state") and seg[-1] == record["state"]:
                    seg[-1] = f"{seg[-1]} {val}"
                else:
                    seg.append(val)
            joined_value = ", ".join([street, *seg])
            text = f"{label}: {joined_value}"
            v = dict(a.value) if isinstance(a.value, dict) else {}
            _note_joined(v, str(a.raw_text or ""), "head")
            for p in pieces:
                _note_joined(v, str(atoms[p].raw_text or ""), "piece")
            v["form_record"] = {**dict(v.get("form_record") or {}), **record,
                                "full_address": joined_value}
            try:
                a = a.model_copy(update={"raw_text": text, "normalized_text": normalize_text(text),
                                         "value": v})
            except Exception:
                pass
        out.append(a)
    # A piece with no address to join keeps its own line: only absorbed ones go.
    return out


def join_contact_phone(atoms: list[Any]) -> list[Any]:
    """A form with exactly ONE contact field and one phone / email field on a
    page: the phone and email are that contact's. Any second contact label on
    the page (an alternate contact, a "Contact | Phone" grid row) and nothing
    is joined."""
    fields = [_field(a.raw_text) for a in atoms]
    pages: dict[Any, dict[str, list[int]]] = {}
    for i, a in enumerate(atoms):
        t = str(a.raw_text or "").lower()
        page = pages.setdefault(_page(a), {"contact": [], "phone": [], "email": [], "other_contact": []})
        f = fields[i]
        if f and f[1] in _CONTACT_LABELS:
            page["contact"].append(i)
        elif f and f[1] in _PHONE_LABELS:
            page["phone"].append(i)
        elif f and f[1] in _EMAIL_LABELS:
            page["email"].append(i)
        elif "contact" in t:
            page["other_contact"].append(i)
    absorbed: dict[int, list[int]] = {}
    for page in pages.values():
        if len(page["contact"]) != 1 or page["other_contact"]:
            continue
        c = page["contact"][0]
        if not fields[c][2] or _NOISE.match(fields[c][2]):
            continue
        for kind in ("phone", "email"):
            if len(page[kind]) == 1:
                p = page[kind][0]
                if fields[p][2] and not _NOISE.match(fields[p][2]):
                    absorbed.setdefault(c, []).append(p)
    if not absorbed:
        return atoms
    gone = {p for ps in absorbed.values() for p in ps}
    out: list[Any] = []
    for i, a in enumerate(atoms):
        if i in gone:
            continue
        if i in absorbed:
            label, _k, name = fields[i]
            extra = [f"{fields[p][0].strip()} {fields[p][2]}" for p in absorbed[i]]
            text = f"{label}: {name}, " + ", ".join(extra)
            v = dict(a.value) if isinstance(a.value, dict) else {}
            _note_joined(v, str(a.raw_text or ""), "head")
            rec = dict(v.get("form_record") or {})
            rec["contact"] = name
            for p in absorbed[i]:
                _note_joined(v, str(atoms[p].raw_text or ""), "piece")
                key = "phone" if fields[p][1] in _PHONE_LABELS else "email"
                rec[key] = fields[p][2]
            v["form_record"] = rec
            try:
                a = a.model_copy(update={"raw_text": text, "normalized_text": normalize_text(text),
                                         "value": v})
            except Exception:
                pass
        out.append(a)
    return out


def join_form_records(atoms: list[Any]) -> list[Any]:
    """Duplicates out, then address pieces and a lone contact's phone in."""
    try:
        atoms = drop_duplicate_form_answers(atoms)
        atoms = join_address_pieces(atoms)
        atoms = join_contact_phone(atoms)
    except Exception:  # pragma: no cover - a join never fails the parse
        import logging
        logging.getLogger(__name__).debug("form record join skipped", exc_info=True)
    return atoms
