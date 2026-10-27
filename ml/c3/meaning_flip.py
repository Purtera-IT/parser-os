"""Minimal meaning edits with a certain new answer, for the meaning_flip loss (losses.meaning_flip_loss).

The contrast test (~/c3-local/runE/contrast_gate.py, the sensitivity half of gate 2) changes the ONE word
that carries a labeled line's meaning and checks that the answer moves. Pilot C stayed put: the supplier
party swapped 1/13 and 0/9, an explicit exclusion 3%, a street added to a city/zip row 0/46. entity_swap,
layout_lock and keyword_dropout only ever teach "the surface changed, keep the answer"; nothing taught
"the meaning changed, move". This module makes those edits on training lines so the heads learn the logic.

Each edit starts from a line whose SAVED gold (the batch's targets, never the model's reading) matches the
family's pre-condition, and its new answer follows from the head's own written definition
(app/core/atom_types.json via the schema answers), so it is certain:

  sup_to_cust   col:supplier us, the line's subject is ours ("We / PurTera / The provider will ...")
                -> that subject becomes the end client                                       -> customer
  sup_to_us     col:supplier customer, the subject is the customer ("Customer to / will ...")
                -> that subject becomes our crew                                             -> us
                (Supplier = "who supplies, does or owns what the line states". Only the subject party is
                swapped, and the line must name no other party before or after.)
  addr_street   read:address_level city_only, the text is a bare "City, ST" (or "City, Country")
                -> a street put in front                                                     -> street_no_zip
                -> a street in front and a ZIP after a US state                              -> full
                city_only table row with a city cell then a ZIP cell
                -> a separate street cell before the city                                    -> full
                city_only roster row starting with a city cell, no ZIP
                -> a street cell before the city                                             -> street_no_zip
                -> that, and a ZIP cell after the city                                       -> full
                ("full: a street address with city and zip"; "street_no_zip: a street address and city, but
                the zip code is missing"). The other direction (a full address losing its street or ZIP) has
                8 qualifying training lines in 7 deals: too few to matter, so it is not a family.
  excl_wrap     col:label_type task/work_scope_item/scope_item/deliverable, admission keep, no negation in
                the line -> wrapped as an explicit leave-out    -> col:label_type exclusion, read:scope_facet
                exclusion ("Work or items the job explicitly leaves out"; "Something the job leaves out")
  adm_to_talk   col:admission keep on a scope fact -> replaced by pure relationship talk
                                                         -> col:admission drop, col:label_type small_talk
  adm_add_fact  col:label_type small_talk with col:admission drop -> one job fact put in front
                -> col:admission keep ("If the line ALSO says something about the job ... it is not small talk")

HELD OUT, on purpose:
  * Every inserted phrase, template, street and fact is written here and is disjoint from contrast_gate.py's
    (tests/test_c3_meaning_flip.py asserts it), and where the gate edits one way this file edits another
    (suffix and wrapper exclusions, not the gate's "Out of scope: " prefixes; a separate street cell, not
    a street put into the city cell; the fact before the small talk, not "Also, ..." after it), so the gate
    still measures generalization.
  * No text is drawn from any deal other than the training line being edited, and a part of a held-out,
    validation or selection deal is never edited (``held_out``).
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

#: Edited lines per family per part.
CAP = 32

#: The deals the contrast gate and the other gates read: never edited, whatever the training dirs hold.
#: (The fixed split in ml/c3/holdout.py is added when it loads.)
HELD_OUT = frozenset("000026 010269 010155 010358 010279 010038 010387 010043 000019 010035 000067 010447".split())

SUPPLIER, ADDRESS, TYPE, FACET, ADMISSION = ("col:supplier", "read:address_level", "col:label_type",
                                             "read:scope_facet", "col:admission")
HEADS = (SUPPLIER, ADDRESS, TYPE, FACET, ADMISSION)
WORK_TYPES = frozenset({"task", "work_scope_item", "scope_item", "deliverable"})
SCOPE_TYPES = WORK_TYPES | {"dependency", "requirement", "assumption", "exclusion", "service_line",
                            "acceptance_criterion"}
FAMILIES = ("sup_to_cust", "sup_to_us", "addr_street", "excl_wrap", "adm_to_talk", "adm_add_fact")
#: C3_FLIP_SKIP="addr_street,...": families left out of training (off by default; every family trains).
SKIP = frozenset(x.strip() for x in os.environ.get("C3_FLIP_SKIP", "").split(",") if x.strip())

# ---------------------------------------------------------------- inserted text (disjoint from the gate's)
#: Our side, singular so "is / will / to" stay grammatical.
OUR_SUBJECTS = ("Our field team", "Our install crew", "Our project team", "Our onsite crew")
#: The end customer's side.
CLIENT_SUBJECTS = ("The end client", "The end client's IT staff", "The end client's facilities team",
                   "The building owner's IT group")
STREETS = ("Larkspur Lane", "Quarry Road", "Fenwick Court", "Halvorsen Parkway", "Tamarack Way",
           "Brookhollow Drive", "Saddleback Road", "Wexford Avenue")
EXCLUSION_TEMPLATES = ("{body} (omitted from this quote)",
                       "{body} -- this work is not part of the proposal.",
                       "This proposal does not cover the following: {body}",
                       "{body}; this is outside the quoted work.")
SMALL_TALK = ("Cheers, and thanks for looping me in!", "Hope the trip back went smoothly.",
              "Thanks a ton, enjoy your evening!", "Good to hear from you, hope all is well on your end.",
              "Much appreciated, have a good one.", "Thanks for the update, chat with you Monday.",
              "No rush at all, I appreciate you getting back to me.")
JOB_FACTS = ("Separately, the quote should cover 3 technicians for 2 nights to swap the core switches.",
             "One more item: we need 40 ceiling speakers wired and tested in the warehouse.",
             "Quick add: the site needs 6 badge readers installed on the loading dock doors.",
             "Before I forget, please price 16 hours of after-hours cutover labor.",
             "Note that the scope now adds 2 new wireless bridges between the buildings.")

#: Every inserted string, for the disjointness test against contrast_gate.py.
TEMPLATES = (OUR_SUBJECTS + CLIENT_SUBJECTS + STREETS + EXCLUSION_TEMPLATES + SMALL_TALK
             + JOB_FACTS)


@dataclass(frozen=True)
class Flip:
    """Line ``line`` of the part read as ``text`` must answer ``targets`` (head key -> answer value)."""
    line: int
    family: str
    text: str
    targets: dict[str, str] = field(default_factory=dict)


#: A negation anywhere: an exclusion or a refusal (the supplier of an exclusion is the side that wrote it).
NEGATION = re.compile(r"\b(not|no|never|without|none|nor|excluded?|excludes|excluding|exclusions?|omit\w*|"
                      r"out\s+of\s+scope|outside|by\s+others|n['’]t|unless|except\w*)\b", re.I)

# ---------------------------------------------------------------- supplier
LEAD = r"(?:^|(?<=[.!?;]\s)|(?<=^\d[.)]\s)|(?<=^\d\d[.)]\s)|(?<=^[-*•]\s))"
OURS = r"(?:PurTera(?:[- ]IT)?|Purtera(?:[- ]IT)?|PURTERA(?:[- ]IT)?|We|(?:The\s+)?(?:[Ss]ervice\s+)?[Pp]rovider)"
THEIRS = r"(?:(?:The\s+)?(?:[Cc]ustomer|CUSTOMER|[Cc]lient|CLIENT))"
DUTY = (r"(?P<duty>will\s+be\s+responsible\s+for|(?:is|are)\s+responsible\s+for|agrees?\s+to|(?:is|are)\s+to|"
        r"will\s+need\s+to|needs?\s+to|will|shall|must|to)\s+(?P<verb>[a-z]+)")
#: A verb after the duty that does not make the subject the one who supplies or does it.
NOT_DOING = frozenset({"be", "been", "receive", "pay", "get", "have", "need", "decide", "approve", "sign",
                       "accept", "review", "confirm", "let", "know", "want", "invoice", "bill", "charge", "not"})
US_SUBJ = re.compile(LEAD + rf"(?P<who>{OURS})\s+" + DUTY)
CUST_SUBJ = re.compile(LEAD + rf"(?P<who>{THEIRS})\s+" + DUTY)
MENTIONS_US = re.compile(r"\b(purtera\w*|provider|we|us|our|ours|ourselves)\b", re.I)
MENTIONS_CUST = re.compile(r"\b(customer'?s?|client'?s?|end[- ]?users?|owner'?s?|property\s+manage\w*|you|your)\b", re.I)
#: Another party, or people whose side the line leaves open ("the engineer", "a technician"): after the swap
#: the supplier would no longer be certain.
OTHER_PARTY = re.compile(r"\b(partner|reseller|cdw|shi|third[- ]part\w*|by others|subcontract\w*|vendor|"
                         r"contractor|manufacturer|carrier|landlord|installer|integrator|electrician|distributor|"
                         r"engineers?|technicians?|techs?|crews?|teams?|staff|seller'?s?|buyer'?s?)\b", re.I)


#: "<items> will be provided by the customer": the party after a supplying participle.
BY_US = re.compile(r"\b(?P<pp>provided|supplied|furnished|delivered|performed|completed|installed|handled|"
                   r"procured|shipped)\s+by\s+(?P<who>PurTera(?:[- ]IT)?|Purtera(?:[- ]IT)?|PURTERA(?:[- ]IT)?|"
                   r"(?:the\s+)?(?:service\s+)?provider|us)\b", re.I)
BY_CUST = re.compile(r"\b(?P<pp>provided|supplied|furnished|delivered|performed|completed|installed|handled|"
                     r"procured|shipped)\s+by\s+(?P<who>(?:the\s+)?(?:customer|client))\b", re.I)


def _lower_first(phrase: str) -> str:
    return phrase[:1].lower() + phrase[1:]


def _swap_party(text: str, subj: re.Pattern, by: re.Pattern, phrases: Sequence[str],
                rng: random.Random) -> str | None:
    """Every phrase naming the party that does or supplies something (``subj``: the sentence's subject before
    a duty; ``by``: after a supplying participle) becomes one of ``phrases``. None when nothing matches or a
    verb after the duty does not make the subject the doer."""
    phrase = rng.choice(list(phrases))
    hits = list(subj.finditer(text))
    if any(m.group("verb") in NOT_DOING for m in hits):
        return None
    out = text
    for m in reversed(hits):
        duty = m.group("duty")
        if re.match(r"are\b", duty):   # "We are to / are responsible" -> one party: "is"
            duty = "is" + duty[3:]
        out = out[:m.start("who")] + phrase + " " + duty + " " + m.group("verb") + out[m.end("verb"):]
    passive = list(by.finditer(out))
    for m in reversed(passive):
        out = out[:m.start("who")] + _lower_first(phrase) + out[m.end("who"):]
    return out if hits or passive else None


def edit_sup_to_cust(text: str, rng: random.Random) -> tuple[str, dict[str, str]] | None:
    if MENTIONS_CUST.search(text) or OTHER_PARTY.search(text) or NEGATION.search(text):
        return None
    out = _swap_party(text, US_SUBJ, BY_US, CLIENT_SUBJECTS, rng)
    if out is None or MENTIONS_US.search(out):
        return None
    return out, {SUPPLIER: "customer"}


def edit_sup_to_us(text: str, rng: random.Random) -> tuple[str, dict[str, str]] | None:
    if MENTIONS_US.search(text) or OTHER_PARTY.search(text) or NEGATION.search(text):
        return None
    out = _swap_party(text, CUST_SUBJ, BY_CUST, OUR_SUBJECTS, rng)
    if out is None or MENTIONS_CUST.search(out):
        return None
    return out, {SUPPLIER: "us"}


# ---------------------------------------------------------------- address level
US_STATES = frozenset("""AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ
NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC""".split())
#: A bare "City, ST" / "City, Country" line: nothing but a place.
CITY_LINE = re.compile(r"^\s*(?P<city>[A-Z][A-Za-z.'-]+(?: [A-Z][A-Za-z.'-]+){0,3}),\s*"
                       r"(?P<region>[A-Z]{2}|[A-Z][a-z]{2,}(?: [A-Z][a-z]+)?)\s*$")
STREETISH = re.compile(r"\b\d{1,6}[A-Za-z]?\s+(?:[NSEW]\.?\s+)?[A-Za-z0-9.' -]{1,40}?\b(?:Street|St|Avenue|Ave|Road|Rd|"
                       r"Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Parkway|Pkwy|Court|Ct|Highway|Hwy|Place|Pl|Circle|Cir|"
                       r"Trail|Trl|Terrace|Pike|Plaza|Loop|Square|Sq)\b\.?", re.I)
ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
def _cells(text: str) -> list[str]:
    return [c.strip() for c in text.split("|")]


def _street(rng: random.Random) -> str:
    return f"{rng.randint(20, 9899)} {rng.choice(STREETS)}"


#: A cell that already reads like a street without a suffix ("Danvers 35 Cherry Hill"): leave the row alone.
NUMBER_THEN_NAME = re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+")
CITY_CELL = re.compile(r"[A-Z][a-z][A-Za-z.'-]*(?: [A-Z][a-z][A-Za-z.'-]*){0,2}")


def edit_addr_street(text: str, rng: random.Random) -> tuple[str, dict[str, str]] | None:
    """A city-only line gets a street:
    * a bare "City, ST" -> "<street>, City, ST" (street_no_zip), or with a ZIP after a US state (full);
    * a row with a city cell then a ZIP cell -> a separate street cell before the city (full);
    * a roster row that starts with a city cell and has no ZIP -> a street cell before the city
      (street_no_zip), or that and a ZIP cell after the city (full)."""
    if STREETISH.search(text) or "\n" in text:
        return None
    m = CITY_LINE.match(text)
    if m:
        lead = f"{_street(rng)}, {m.group('city')}, {m.group('region')}"
        if m.group("region") in US_STATES and rng.random() < 0.5:
            return f"{lead} {rng.randint(10000, 99899):05d}", {ADDRESS: "full"}
        return lead, {ADDRESS: "street_no_zip"}
    if "|" not in text or NUMBER_THEN_NAME.search(text):
        return None
    cells = _cells(text)
    zips = ZIP.findall(text)
    for i in range(len(cells) - 1):
        city, z = cells[i], cells[i + 1]
        if (CITY_CELL.fullmatch(city) and re.fullmatch(r"\d{5}(?:-\d{4})?", z) and len(zips) == 1):
            new = cells[:i] + [_street(rng)] + cells[i:]
            return " | ".join(new), {ADDRESS: "full"}
    if not zips and len(cells) >= 3 and CITY_CELL.fullmatch(cells[0]) and ":" not in text:
        new = [_street(rng), cells[0]]
        if rng.random() < 0.5:
            return " | ".join(new + [f"{rng.randint(10000, 99899):05d}"] + cells[1:]), {ADDRESS: "full"}
        return " | ".join(new + cells[1:]), {ADDRESS: "street_no_zip"}
    return None


# ---------------------------------------------------------------- exclusion
MARKER = re.compile(r"^\s*(?:\d{1,3}[.)]|\(\d{1,3}\)|[-*•])\s+")
#: A request, a pointer or someone speaking is not a work item to leave out; the templates name the quote.
NOT_AN_ITEM = re.compile(r"\b(i|i'm|i'll|we|we're|we'll|you|your|please|quote|proposal|attached|see|review|"
                         r"following|let|thanks?)\b|\?", re.I)


def edit_excl_wrap(text: str, rng: random.Random) -> tuple[str, dict[str, str]] | None:
    t = text.strip()
    if (not t or NEGATION.search(t) or NOT_AN_ITEM.search(t) or "\n" in t or t.count("|") >= 2 or "\t" in t or len(t) > 300
            or len(t.split()) < 3 or t.endswith("?")):
        return None
    m = MARKER.match(t)
    lead, body = (t[:m.end()], t[m.end():]) if m else ("", t)
    body = body.rstrip(" .;:,")
    out = lead + rng.choice(EXCLUSION_TEMPLATES).format(body=body)
    return out, {TYPE: "exclusion", FACET: "exclusion"}


# ---------------------------------------------------------------- admission
def edit_adm_to_talk(text: str, rng: random.Random) -> tuple[str, dict[str, str]] | None:
    if not text.strip():
        return None
    return rng.choice(SMALL_TALK), {ADMISSION: "drop", TYPE: "small_talk"}


def edit_adm_add_fact(text: str, rng: random.Random) -> tuple[str, dict[str, str]] | None:
    t = text.strip()
    if not t or len(t) > 300:
        return None
    return f"{rng.choice(JOB_FACTS)} {t}", {ADMISSION: "keep"}


# ---------------------------------------------------------------- which lines
def eligible(family: str, g: Mapping[str, str | None]) -> bool:
    """The family's pre-condition on the line's SAVED gold (head key -> answer value, None = unlabeled)."""
    t, adm = g.get(TYPE), g.get(ADMISSION)
    if family == "sup_to_cust":
        return g.get(SUPPLIER) == "us"
    if family == "sup_to_us":
        return g.get(SUPPLIER) == "customer"
    if family == "addr_street":
        return g.get(ADDRESS) == "city_only"
    if family == "excl_wrap":
        return t in WORK_TYPES and adm == "keep" and g.get(FACET) != "exclusion"
    if family == "adm_to_talk":
        return adm == "keep" and t in SCOPE_TYPES
    if family == "adm_add_fact":
        return t == "small_talk" and adm == "drop"
    return False


EDITS = {"sup_to_cust": edit_sup_to_cust, "sup_to_us": edit_sup_to_us, "addr_street": edit_addr_street,
         "excl_wrap": edit_excl_wrap, "adm_to_talk": edit_adm_to_talk,
         "adm_add_fact": edit_adm_add_fact}


def candidates(texts: Sequence[str], gold: Mapping[str, Sequence[str | None]],
               rng: random.Random) -> dict[str, list[tuple[int, str, dict[str, str]]]]:
    """Every line each family could edit: {family: [(line, new text, targets)]}."""
    n = len(texts)
    rows = [{k: v[i] for k, v in gold.items() if i < len(v)} for i in range(n)]
    out: dict[str, list] = {}
    for fam, fn in EDITS.items():
        got = []
        for i, t in enumerate(texts):
            if not t or not eligible(fam, rows[i]):
                continue
            e = fn(t, rng)
            if e and e[0] != t:
                got.append((i, e[0], e[1]))
        out[fam] = got
    return out


def plan_flips(texts: Sequence[str], gold: Mapping[str, Sequence[str | None]], rng: random.Random,
               cap: int = CAP) -> list[Flip]:
    """At most ``cap`` edits per family, drawn by ``rng``; each line edited once, never next to another
    edited line (its neighbours are read as written)."""
    cands = candidates(texts, gold, rng)
    fams = [f for f in FAMILIES if f not in SKIP]
    rng.shuffle(fams)
    used: set[int] = set()
    flips: list[Flip] = []
    for fam in fams:
        pool = list(cands.get(fam) or [])
        rng.shuffle(pool)
        taken = 0
        for i, new, targets in pool:
            if taken >= cap:
                break
            if i in used or i - 1 in used or i + 1 in used:
                continue
            used.add(i)
            flips.append(Flip(i, fam, new, dict(targets)))
            taken += 1
    return sorted(flips, key=lambda f: f.line)


def held_out(deal_id: str) -> bool:
    """A part of a gate, validation or selection deal (never edited)."""
    from .holdout import deal_number  # noqa: PLC0415

    code = deal_number(deal_id).partition("@")[0]
    try:
        from .holdout import FIXED_HOLDOUT  # noqa: PLC0415
    except Exception:  # noqa: BLE001 (a host without the split manifest still guards the listed deals)
        FIXED_HOLDOUT = frozenset()
    return code in HELD_OUT or code in FIXED_HOLDOUT


def gold_of(batch, schema) -> dict[str, list[str | None]]:
    """The saved answer per family head per line, read from the batch's targets (never a model's reading):
    head key -> [answer value or None]."""
    from .data import IGNORE  # noqa: PLC0415

    by_key = schema.by_key()
    out: dict[str, list[str | None]] = {}
    for key in HEADS:
        opp, ys = by_key.get(key), batch.targets.get(key)
        if opp is None or ys is None or opp.multiple:
            continue
        out[key] = [opp.answers[y].value if y != IGNORE and 0 <= y < len(opp.answers) else None for y in ys]
    return out
