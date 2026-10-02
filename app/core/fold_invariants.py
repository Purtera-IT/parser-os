"""The two rules that sit above every collapse in the compile.

A dozen stages fold atoms together: `collapse_duplicate_atoms`,
`cross_type_dedup_atoms`, `semantic_dedup_atoms`, `_suppress_table_row_blob_doubles`,
`merge_signature_rows`, `dedupe_stakeholder_atoms`, the rollups. Each one has its
own notion of what a duplicate is, and each one is right about its own case.

What none of them had was a shared floor. So the same defect was found and
fixed eight separate times, in eight separate shapes:

  phase 1-2  81 per-plate cable quantities folded onto one plate's copy
             39 folds where the numbers differed, one of them a bid deadline
             a location's quantities deleted by a twin from another location
  phase 3    a totals block reduced to its revenue line, losing cost, margin
             and margin percentage
             four HubSpot notes and seven message headers each collapsed to one
             a `quantity` retyped into prose that did not state the number
             a contract row on a signature page deleted as a signature
             the same row deleted again, one stage later, as a "duplicate" of
             the signature record that replaced it

Every one is the same sentence: **a fold may not delete what only the loser
held.** That rule was already written down -- it is the promotion gate on the
suppression head in `_BETS_LEDGER.md`, which says a learned head "may never
fold atoms that state different numbers, or that belong to different places."
It was a gate on a head that does not exist yet, while the code re-derived it
by hand at each fold site and missed one every time.

This module is that rule, once.

Two invariants
--------------
**Different numbers are different facts.** Formatting is not a number:
"1,200" and "1200" are one quantity, and so are "2.50" and "2.5", so folding
them must still be allowed. A figure the loser states and the survivor does
not is a deletion.

**Different identities are different facts.** Two atoms with the same words
about different plates, sites, rooms, notes or messages are two facts. Identity
is read from the atom's structured value and its entity keys, never guessed
from its text.

What this module does NOT decide
--------------------------------
Whether a given stage should refuse a fold, merge the loser in, or keep both.
That is policy and it differs per stage -- `cross_type_dedup_atoms` exempts a
money column whose value already lives in the winner, because there the loser's
figure is not lost. This module answers only the factual question each of those
policies needs: *what would this fold delete?*
"""
from __future__ import annotations

import json as _json
import re
from typing import Any

#: A figure, with its thousands separators and decimal part attached.
FIGURE_RE = re.compile(r"\d+(?:[.,]\d+)*")

#: Value keys that say WHICH thing an atom is about. Two atoms with the same
#: words and different values here are about different things.
IDENTITY_FIELDS: tuple[str, ...] = (
    "plate_id",
    "site",
    "site_key",
    "room",
    "location",
    # Added in phase 3: a HubSpot note and an email message are identities in
    # exactly the same sense, and keying them on a constant collapsed every
    # note on a deal into one.
    "hubspot_note_id",
    "message_id",
)


def _canonical(raw: str) -> str:
    """One figure, with formatting removed and the quantity preserved."""
    cleaned = raw.replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    return str(int(value)) if value.is_integer() else str(value)


def figures_in_order(text: str) -> tuple[str, ...]:
    """Every figure in the text, in order. Order matters to a bucket key."""
    return tuple(_canonical(m) for m in FIGURE_RE.findall(text or ""))


def figures(text: str) -> frozenset[str]:
    """Every figure in the text, as a set. Order does not matter to a guard."""
    return frozenset(figures_in_order(text))


def figures_stated(atom: Any) -> frozenset[str]:
    """Every figure the atom states, in its words OR its structured value.

    Both halves matter, and the second half is the one that gets forgotten. A
    raw table row's trailing "| $21,560.00" is a money column whose amount
    already lives in the typed sibling's `value`; the typed atom is not poorer
    for lacking it in text, and folding the row away loses nothing. The same
    fold becomes a deletion the moment the survivor's value does not carry the
    figure either.
    """
    blob = str(
        getattr(atom, "raw_text", None) or getattr(atom, "text", None) or ""
    )
    value = getattr(atom, "value", None)
    if isinstance(value, dict):
        try:
            blob = blob + " " + _json.dumps(value, default=str)
        except Exception:
            blob = blob + " " + str(value)
    return figures(blob)


_EMAIL_RE = __import__("re").compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def emails_stated(atom: Any) -> frozenset[str]:
    """Every email address the atom states, in its words OR its value.

    An address is how a contact is reached, and it carries no digit, so
    :func:`figures_stated` never protected it: a contact row "Jane Roe | PM |
    jane@acme.com" and its bare twin "Jane Roe | PM" key alike once the key
    is cut, and the row with the address was the one folded away.
    """
    blob = str(getattr(atom, "raw_text", None) or getattr(atom, "text", None) or "")
    value = getattr(atom, "value", None)
    if isinstance(value, dict):
        try:
            blob = blob + " " + _json.dumps(value, default=str)
        except Exception:
            blob = blob + " " + str(value)
    return frozenset(m.group(0).lower().rstrip(".") for m in _EMAIL_RE.finditer(blob))


def emails_only_the_loser_states(winner: Any, loser: Any) -> frozenset[str]:
    """Addresses that leave the compile if ``loser`` is folded into ``winner``."""
    return emails_stated(loser) - emails_stated(winner)


#: Verbs that make a line an INSTRUCTION -- something a person is told to do.
#: A person record never carries one; a SOW step that names a person does
#: ("Call Client Support Manager John Ozuna-Diaz ... upon arrival", Ox 010353),
#: and keying both on the name folded the step into the record and lost it.
#: A closed list rather than a suffix rule, for the reason
#: `semantic_dedup._STATEMENT_VERBS` gives: "Mobis" ends in -s and is a company.
ACTION_VERBS: frozenset[str] = frozenset({
    # Only words that are verbs first. "Phone", "Email", "Text", "Page",
    # "Report" and "Request" head table columns ("Phone Number | 404-...")
    # far more often than they open an instruction.
    "call", "contact", "notify", "inform", "ask", "meet", "check", "arrive",
    "escort", "wait", "bring", "send", "submit", "schedule", "confirm",
    "coordinate", "provide", "install", "remove", "deliver", "ensure",
    "verify", "obtain", "reach", "dial",
})

_WORD_RE = re.compile(r"[a-z][a-z'-]*")

#: A verb is an instruction only where an instruction's verb stands: opening
#: the line or a clause, or after a modal / "please" / "then". "Phone: 555..."
#: and "Contact: Megan" are labels (a colon follows); "Phone" mid-row is a noun.
_ACTION_RE = re.compile(
    r"(?:^|[.;!?\n\u2022*>]\s*|(?:^|\s)[-\u2013\u2014]\s+|\d[.)]\s+"
    r"|\b(?:please|then|and|to|must|should|shall|will|first)\s+)"
    r"(?P<verb>" + "|".join(sorted(ACTION_VERBS, key=len, reverse=True)) + r")\b(?!\s*:)(?=\s+[A-Za-z])",
    re.I,
)


def _words(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall((text or "").lower()))


def _atom_text(atom: Any) -> str:
    return str(getattr(atom, "raw_text", None) or getattr(atom, "text", None) or "")


def actions_stated(atom: Any) -> frozenset[str]:
    """Instruction verbs the atom's words state. Text only: a structured
    value's keys ("email", "phone") are field names, not instructions."""
    return frozenset(m.group("verb").lower() for m in _ACTION_RE.finditer(_atom_text(atom).strip()))


def actions_only_the_loser_states(winner: Any, loser: Any) -> frozenset[str]:
    """Instruction verbs that leave the compile if ``loser`` is folded away."""
    # Lost only when the survivor's words do not carry the verb at all.
    return actions_stated(loser) - _words(_atom_text(winner))


#: A ZIP is five digits after a state: "OH 45840", "Ohio 45840-1234". A bare
#: five-digit number is not one -- street numbers ("15733 US-224") are too,
#: and so is "Suite 12345".
def _zip_re() -> "re.Pattern[str]":
    from app.core.address_parse import US_STATE_NAMES, US_STATES

    states = sorted({*US_STATES, *(n.title() for n in US_STATE_NAMES)}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(x) for x in states) + r")\.?,?\s+(\d{5})(?:-\d{4})?(?!\d)")


_ZIP_RE = _zip_re()
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")


def _blob(atom: Any) -> str:
    blob = _atom_text(atom)
    value = getattr(atom, "value", None)
    if isinstance(value, dict):
        try:
            blob = blob + " " + _json.dumps(value, default=str)
        except Exception:
            blob = blob + " " + str(value)
    return blob


def zips_stated(atom: Any) -> frozenset[str]:
    """ZIP codes the atom states, in its words or its value (`zip`, `postal_code`)."""
    found = {m.group(1) for m in _ZIP_RE.finditer(_atom_text(atom))}
    value = getattr(atom, "value", None)
    if isinstance(value, dict):
        for k in ("zip", "zip_code", "postal_code", "postcode"):
            m = re.match(r"\s*(\d{5})", str(value.get(k) or ""))
            if m:
                found.add(m.group(1))
        for v in value.values():
            if isinstance(v, str):
                found |= {m.group(1) for m in _ZIP_RE.finditer(v)}
    # A ZIP the atom states anywhere -- a value's `address` string included --
    # is carried; only the five digits matter.
    return frozenset(found)


def phones_stated(atom: Any) -> frozenset[str]:
    """Phone numbers the atom states, as their last ten digits."""
    return frozenset(re.sub(r"\D", "", m.group(0))[-10:] for m in _PHONE_RE.finditer(_blob(atom)))


def detail_only_the_loser_states(winner: Any, loser: Any) -> frozenset[str]:
    """What leaves the compile if ``loser`` is folded into ``winner``.

    The general form of :func:`emails_only_the_loser_states`: a fold may not
    drop the only copy that carries a ZIP, a phone number, an email address,
    or an instruction the survivor lacks (Ox 010353: the one address with its
    ZIP, Megan's contact row with her email and phone, and "Call Client
    Support Manager John Ozuna-Diaz ... upon arrival" folded into John's
    person record -- each the only copy, each gone). Each lost item comes back
    tagged -- ``zip:45840``, ``phone:5552013344``, ``email:megan@acme.com``,
    ``action:call`` -- so a stage can log exactly what it would have deleted.

    These are the details a reader acts on, which is why they are named
    rather than read off every figure: a clipped site id "ATL-WEST-0" states
    a figure its canonical "ATL-WEST-02" does not, and folding it loses
    nothing. Ask AFTER merging the loser's value into the winner: a ZIP or an
    address the merge carried across is no longer lost.
    """
    lost = {f"zip:{z}" for z in zips_stated(loser) - zips_stated(winner)}
    lost |= {f"phone:{p}" for p in phones_stated(loser) - phones_stated(winner)}
    lost |= {f"email:{e}" for e in emails_only_the_loser_states(winner, loser)}
    lost |= {f"action:{v}" for v in actions_only_the_loser_states(winner, loser)}
    return frozenset(lost)


def covers(survivor: Any, other: Any) -> bool:
    """Does ``survivor`` state every ZIP, phone, email and instruction ``other`` does?"""
    return not detail_only_the_loser_states(survivor, other)


#: Words a contact record uses as labels, plus function words. They say
#: nothing about WHO or HOW, so a copy that differs only in them loses nothing.
_PERSON_LABEL_WORDS: frozenset[str] = frozenset({
    "name", "names", "title", "role", "phone", "phones", "email", "emails", "mail",
    "cell", "mobile", "office", "tel", "telephone", "fax", "ext", "number", "num",
    "contact", "contacts", "info", "information", "address", "direct", "work",
    "the", "a", "an", "and", "or", "of", "for", "at", "to", "in", "on", "by",
    "with", "is", "as", "s", "mr", "mrs", "ms", "dr", "stakeholder", "person", "job",
})

#: The loser's own person fields: how IT names and titles the one person the
#: fold is about. "Bernie" for "Bernard", a typo'd surname, "Senior Client
#: Executive" for "Client Executive" are that person said another way, and
#: the fields are unioned onto the survivor anyway.
_OWN_FIELDS: tuple[str, ...] = ("name", "role", "title", "company", "organisation", "organization")

#: A note that a detail is ABSENT is itself a detail a reader acts on
#: ("Danny's phone not provided", 010353): it says not to go looking.
_ABSENCE_NOTE_RE = re.compile(
    r"\b(?:not\s+(?:provided|available|given|listed|known)|n/a|tbd|unknown"
    r"|no\s+(?:phone|email|e-mail|cell|number|contact))\b",
    re.I,
)


def person_words(text: str) -> frozenset[str]:
    text = _EMAIL_RE.sub(" ", text or "")
    text = _PHONE_RE.sub(" ", text)
    return frozenset(
        w for w in re.findall(r"[a-z]+", text.lower())
        if len(w) >= 2 and w not in _PERSON_LABEL_WORDS
    )


def person_detail_only_the_loser_states(winner: Any, loser: Any) -> frozenset[str]:
    """What a PERSON record ``loser`` says that ``winner`` does not, beyond
    the ZIP / phone / email / instruction of :func:`detail_only_the_loser_states`.

    A contact row is rarely only one person's name. Deal 010353's SOW
    "CUSTOMER CONTACTS" row named John Ozuna-Diaz AND Danny, with "Danny's
    phone not provided"; it keyed on John's name, folded into John's bare
    record, and Danny, his role and the note left the compile. Lost here: a
    word of the loser's text (another name, a role, a note) the survivor
    states nowhere in its words or value, and any absence note the survivor
    lacks. Labels ("Phone:", "Name:"), function words and the loser's own
    name / role / title (the one person the fold is about) are not detail.
    Tagged ``word:danny`` / ``note:not provided``. Ask AFTER merging.
    """
    w_text = _atom_text(winner)
    own: frozenset[str] = frozenset()
    value = getattr(loser, "value", None)
    if isinstance(value, dict):
        own = person_words(" ".join(str(value.get(k) or "") for k in _OWN_FIELDS))
    lost = {f"word:{w}" for w in person_words(_atom_text(loser)) - person_words(_blob(winner)) - own}
    have = {" ".join(m.group(0).lower().split()) for m in _ABSENCE_NOTE_RE.finditer(w_text)}
    lost |= {
        f"note:{n}"
        for n in (" ".join(m.group(0).lower().split()) for m in _ABSENCE_NOTE_RE.finditer(_atom_text(loser)))
        if n not in have
    }
    return frozenset(lost)


def identity(atom: Any) -> tuple[str, ...]:
    """What distinguishes this atom from another with the SAME words."""
    ident: list[str] = []
    value = getattr(atom, "value", None)
    if isinstance(value, dict):
        for field in IDENTITY_FIELDS:
            got = value.get(field)
            if got not in (None, "", [], {}):
                ident.append(f"{field}={got}")
    keys = getattr(atom, "entity_keys", None)
    if keys:
        # An atom tied to a different entity is about a different thing, and
        # entity keys are already canonical, so they compare cleanly.
        ident.extend(sorted(str(k) for k in keys))
    return tuple(ident)


# ---- the two questions a fold site actually asks ---------------------------


def figures_only_the_loser_states(winner: Any, loser: Any) -> frozenset[str]:
    """Figures that leave the compile if ``loser`` is folded into ``winner``."""
    return figures_stated(loser) - figures_stated(winner)


def identities_differ(winner: Any, loser: Any) -> bool:
    """True when the two atoms are about demonstrably different things.

    Only a stated identity counts. An atom that records no plate, site, room,
    note or message is not thereby "somewhere else" -- it is simply unplaced,
    and refusing every fold involving one would stop dedup entirely.
    """
    a, b = identity(winner), identity(loser)
    return bool(a) and bool(b) and a != b


def refuse_fold(winner: Any, loser: Any) -> str | None:
    """A reason this fold would delete a fact, or None when it is safe.

    The default floor, for a stage with no exemption of its own. A stage that
    knows better about its own case -- a money column already carried in the
    winner's value, a per-document index that is not a fact -- asks the two
    questions above directly instead.
    """
    lost = figures_only_the_loser_states(winner, loser)
    if lost:
        return "states figures the survivor does not: " + ", ".join(sorted(lost)[:6])
    gone = emails_only_the_loser_states(winner, loser)
    if gone:
        return "states email addresses the survivor does not: " + ", ".join(sorted(gone)[:6])
    acts = actions_only_the_loser_states(winner, loser)
    if acts:
        return "states an instruction the survivor does not: " + ", ".join(sorted(acts)[:6])
    if identities_differ(winner, loser):
        return f"different identity: {identity(loser)} vs {identity(winner)}"
    return None
