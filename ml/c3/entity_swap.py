"""Consistent entity / name swaps of a part's line texts, for the entity_swap loss (losses.entity_swap_loss).

Pilot C failed the gate-2 swap test on 22 of 59 heads; the entity kind moved col:label_type on 11.7% of
the labeled lines and nothing in training ever showed the model a renamed deal. This module makes the
SAME KINDS of edits as the gate's entity perturbation (~/c3-local/runE/swap_gate.py, section (b)), with
the same detectors (copied below, regex for regex) and the same protections, so training targets the
measured failure:

  emails      the local part becomes a first name; the domain becomes a company unless it is ours or a
              generic mailbox (gmail, outlook, ...)
  brands      "<Capitalised words> Inc/LLC/Corp/..." and every non-generic email domain, swapped wherever
              the brand appears in the part
  places      cities, state codes (after ", " / "| " or before a ZIP), ZIP digits after a state, and
              all-caps table cells that are not dictionary words
  people      first names (proper names that are not dictionary words) and the capitalised surname after one
  IDs         letter-prefixed codes (INC0012345, DC1234, SR-12345) and 3+ digits after an ID word
              (site/store/ticket/case/PO/ref/...): digit for digit, same length, same leading-zero rule
  never       our own company ("purtera", the batch's company) and its domain; counts, hours, prices,
              dates, quantities and every other bare number; every lowercase word

Every entity keeps ONE replacement across the part, as the gate keeps one across the deal.

What differs from the gate, on purpose:
  * The REPLACEMENT pools (first / last names, cities, states, companies) are disjoint from the gate's, so
    gate 2 stays a held-out measurement: training never shows the model the gate's substitutes.
  * The map is drawn from a seeded ``random.Random`` per step (the gate hashes each entity once), so a
    name never maps to the same stand-in twice in a row and the heads cannot learn the stand-ins.
  * Cities are matched in their capitalised / upper-case spelling only (the gate matches any case, which
    also renames the words "spring", "temple", "mesa" in prose: a meaning cue, not a place).
  * One pass per city regex, so a swapped city is never swapped again (the gate's per-city loop chains).
No substitution is drawn from any deal: the pools are fixed lists in this file. The dictionary words and
proper names in fixtures/entity_swap_words.json only DETECT (they are the gate's /usr/share/dict checks,
vendored for hosts without a dictionary; built from training files only, holdouts and selection excluded:
``python -m ml.c3.entity_swap --build-words <train dirs>``).
"""
from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

WORDS_FILE = Path(__file__).with_name("fixtures") / "entity_swap_words.json"

# ---------------------------------------------------------------- the gate's detectors (swap_gate.py (b))
GENERIC_MAIL = ("gmail", "outlook", "yahoo", "hotmail", "icloud", "aol", "live", "msn", "me")
SUFFIX = r"(?:Inc|LLC|Ltd|Corp|Corporation|Co|Group|Solutions|Technologies|Systems|Partners|Services)\b\.?"
EMAIL = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+)((?:\.[A-Za-z0-9-]+)+)\b")
COMPANY = re.compile(rf"\b((?:[A-Z][\w&'-]*\s+){{0,3}}[A-Z][\w&'-]*)\s+({SUFFIX})")
CELL_CAPS = re.compile(r"((?:^|\|\s*|\t|:\s+))([A-Z][A-Z'-]{3,}(?: [A-Z][A-Z'-]{2,})?)(?=\s*(?:\||\t|$))")
TITLE = re.compile(r"\b[A-Z][a-z][a-zA-Z'-]+\b")
#: The gate's detected states (its pool). Replacements come from TRAIN_STATES.
GATE_STATES = ["AL", "AZ", "CA", "CO", "FL", "GA", "IL", "IN", "KY", "LA", "MA", "MD", "MI", "MN", "MO", "NC",
               "NJ", "NV", "NY", "OH", "OK", "OR", "PA", "SC", "TN", "TX", "UT", "VA", "WA", "WI"]
#: The gate's state pattern, with one meaning guard: after ", " / "| " the code must end the cell or clause
#: ("..., TX", "| TX |"), so "SECTION 20, IN NO EVENT" keeps its IN.
STATE = re.compile(r"(?<=[,|]\s)(%s)\b(?=\s*(?:$|[|\t,;)]|\d{5}\b))|\b(%s)(?=\s+\d{5}\b)"
                   % ("|".join(GATE_STATES), "|".join(GATE_STATES)))
ZIP = re.compile(r"\b(%s)(\s+)(\d{5})(-\d{4})?\b" % "|".join(GATE_STATES))
ID_TOKEN = re.compile(r"\b([A-Z]{2,6})(-?)(\d{4,})\b")
#: Meaning guard: "Checkout LLC-2026.07" is a company and a year, not an ID.
NOT_ID_PREFIX = frozenset({"LLC", "LTD", "CORP", "CO", "FY", "Q"})
ID_AFTER = re.compile(r"\b(site|store|loc|ticket|case|po|ref|req|ritm|inc|wo|sr|dc|id|acct|account|asset|tag|"
                      r"serial|s/n|sn)(\s*(?:#|no\.?|id|number)?\s*[:#-]?\s*)(\d{3,})\b", re.I)
#: The gate's city list: detected here, never used as a replacement.
GATE_CITIES = ["Abilene", "Akron", "Albany", "Amarillo", "Anaheim", "Arlington", "Atlanta", "Austin", "Bakersfield",
               "Baltimore", "Baton Rouge", "Beaumont", "Boise", "Boston", "Bryan", "Buffalo", "Chandler",
               "Charlotte", "Chicago", "Cleveland", "Columbus", "Conroe", "Corpus Christi", "Dallas", "Dayton",
               "Denton", "Denver", "Detroit", "El Paso", "Fresno", "Frisco", "Garland", "Houston", "Irving", "Katy",
               "Killeen", "Laredo", "Lubbock", "Madison", "McAllen", "Memphis", "Mesa", "Miami", "Midland",
               "Nashville", "Odessa", "Omaha", "Orlando", "Pasadena", "Pearland", "Phoenix", "Plano", "Portland",
               "Raleigh", "Reno", "Richmond", "Sacramento", "San Antonio", "Spokane", "Spring", "Sugar Land",
               "Tampa", "Temple", "Tomball", "Tucson", "Tulsa", "Tyler", "Waco", "Wichita", "Woodlands"]
#: The gate's replacement pools, kept only so a test can prove training never draws from them.
GATE_LAST = ["Alvarez", "Brennan", "Castillo", "Dempsey", "Ellison", "Farrow", "Gallagher", "Hollis", "Iverson",
             "Jarvis", "Kowalski", "Lindqvist", "Moreau", "Nakamura", "Okafor", "Pellegrini", "Quintero",
             "Rasmussen", "Sorensen", "Tremblay", "Underwood", "Vasquez", "Whitaker", "Yamada", "Zielinski"]
GATE_COMPANIES = ["Aldermoor", "Brightwater", "Calloway", "Dunmore", "Evergale", "Fairhaven", "Granitepeak",
                  "Harborline", "Ironbridge", "Juniper", "Kestrel", "Lakemont", "Meridian", "Northvale", "Oakhurst",
                  "Pinecrest", "Quarrystone", "Redwater", "Silverline", "Thornbury", "Upland", "Westbrook"]

# ---------------------------------------------------------------- training's own replacement pools
#: Not in /usr/share/dict/propernames (the gate's first-name pool) and not dictionary words.
TRAIN_FIRST = ["Aiden", "Callum", "Esme", "Farah", "Ines", "Kenji", "Leona", "Mateo", "Nadia", "Orla", "Priya",
               "Saoirse", "Tariq", "Valentina", "Yusuf", "Zara", "Anika", "Emeka", "Jovan", "Kiran", "Lorcan",
               "Mireille", "Nikhil", "Paloma", "Thiago", "Vikram", "Yara", "Zainab", "Ayesha", "Benedikt",
               "Caoimhe", "Dmitri", "Eilidh", "Giulia", "Hiroshi", "Joaquin", "Keziah", "Lucien", "Maeve",
               "Niamh", "Osvaldo", "Perpetua", "Rasmus", "Siobhan", "Vesna", "Wiktor", "Ximena", "Yaroslav",
               "Zofia"]
TRAIN_LAST = ["Abernathy", "Bergstrom", "Cardenas", "Delacroix", "Esposito", "Halvorsen", "Ibarra", "Jablonski",
              "Kaczmarek", "Lachance", "Novak", "Oyelaran", "Petrakis", "Quigley", "Rinaldi", "Szabo", "Takahashi",
              "Urquhart", "Valdivia", "Wojcik", "Yilmaz", "Zamora", "Achterberg", "Constantinou", "Dragomir",
              "Eriksen", "Fonseca", "Grigoryan", "Haddad"]
TRAIN_CITIES = ["Albuquerque", "Anchorage", "Asheville", "Billings", "Birmingham", "Boulder", "Charleston",
                "Chattanooga", "Cheyenne", "Des Moines", "Eugene", "Fargo", "Flagstaff", "Fort Collins",
                "Greenville", "Hartford", "Honolulu", "Huntsville", "Indianapolis", "Knoxville", "Lansing",
                "Lexington", "Little Rock", "Louisville", "Milwaukee", "Missoula", "Montgomery", "Naperville",
                "Newark", "Norfolk", "Olympia", "Peoria", "Pittsburgh", "Providence", "Provo", "Rochester",
                "Savannah", "Scranton", "Seattle", "Sioux Falls", "Springfield", "Syracuse", "Tacoma",
                "Tallahassee", "Topeka", "Trenton", "Wilmington", "Worcester", "Yonkers"]
#: The states the gate never uses, minus the ones that read as words or headers (ID, ME, HI, OK...).
TRAIN_STATES = ["AK", "AR", "CT", "DE", "IA", "KS", "MS", "MT", "ND", "NE", "NH", "NM", "RI", "SD", "VT", "WV",
                "WY"]
TRAIN_COMPANIES = ["Ashgrove", "Bellhaven", "Copperfield", "Elmstead", "Foxhollow", "Glenbrook", "Hawthorne",
                   "Kingsley", "Millbrook", "Ravenswood", "Stonegate", "Valebridge", "Willowmere", "Wrenfield",
                   "Amberlyn", "Brackenridge", "Coldharbor"]

#: Cities detected: the gate's and training's (so a training line naming Seattle is renamed too).
CITIES = sorted(set(GATE_CITIES) | set(TRAIN_CITIES), key=len, reverse=True)
CITY_RE = re.compile(r"\b(" + "|".join(re.escape(c) + "|" + re.escape(c.upper()) for c in CITIES) + r")\b")
CITY_SET = {c.lower() for c in CITIES}


@lru_cache(maxsize=1)
def _lexicon() -> tuple[frozenset[str], frozenset[str]]:
    """(names, dictionary): the gate's FIRST_SET minus dictionary words, and the lowercase dictionary words
    it checks. The vendored list, plus /usr/share/dict when the host has one (closer to the gate on words
    the training files never had)."""
    names: set[str] = set()
    words: set[str] = set()
    try:
        blob = json.loads(WORDS_FILE.read_text())
        names, words = set(blob["names"]), set(blob["dict_words"])
    except (OSError, ValueError, KeyError):
        pass
    try:
        with open("/usr/share/dict/words") as f:
            words |= {w.strip().lower() for w in f if w.strip()[:1].islower()}
    except OSError:
        pass
    return frozenset(n for n in names if n.lower() not in words), frozenset(words)


def _own(s: str, own: set[str]) -> bool:
    s = s.lower()
    return any(o and o in s for o in own)


def _case(src: str, rep: str) -> str:
    return rep.upper() if src.isupper() else rep.lower() if src.islower() else rep


class EntitySwap:
    """One part's swap map: every person, email, brand, place and ID keeps one replacement. Ours never moves."""

    def __init__(self, texts: Iterable[str], company: str | None, rng: random.Random):
        self.rng = rng
        self.own = {"purtera", (company or "").strip().lower()} - {""}
        self.names, self.dict = _lexicon()
        self.map: dict[str, str] = {}
        self.used: dict[str, set[str]] = {}
        brands = set()
        texts = list(texts)
        # Words the part writes in lower case in its prose (emails and web addresses aside) are common words,
        # whatever the dictionary says ("paging", "shared"): never a brand.
        prose = set()
        for t in texts:
            prose.update(re.findall(r"(?<![@./\w])[a-z][a-z'-]+(?![@\w]|\.[a-z])", EMAIL.sub(" ", t or "")))
        for t in texts:
            for m in EMAIL.finditer(t or ""):
                d = m.group(2).lower()
                if d not in GENERIC_MAIL and not _own(d, self.own):
                    brands.add(d)
            for m in COMPANY.finditer(t or ""):
                b = " ".join(m.group(1).split()).lower()
                if not _own(b, self.own):
                    brands.add(b)
        # Meaning guards: a "brand" with digits is a code glued to a name (PR0000033245-Checkout LLC), and a
        # brand whose every word the part also writes in lower case ("Professional Services", "Paging
        # Systems") would rename those words in every line ("some professional services"): the gate does,
        # training does not. "Acme Corp" stays a brand while nobody writes "acme".
        self.brands = {b for b in brands if len(b) >= 3 and not _own(b, self.own)
                       and not any(c.isdigit() for c in b)
                       and not all(w in prose for w in re.split(r"[\s&'-]+", b) if w)}
        self.brand_map = {b: self._pick("brand", b, TRAIN_COMPANIES) for b in sorted(self.brands)}
        self.brand_re = (re.compile(r"\b(" + "|".join(map(re.escape, sorted(self.brands, key=len, reverse=True)))
                                    + r")\b", re.I) if self.brands else None)

    def _pick(self, kind: str, src: str, pool) -> str:
        used = self.used.setdefault(kind, set())
        free = [c for c in pool if c.lower() != src.lower() and c.lower() not in used]
        c = self.rng.choice(free or [c for c in pool if c.lower() != src.lower()] or list(pool))
        used.add(c.lower())
        return c

    def _get(self, kind: str, src: str, pool) -> str:
        key = f"{kind}:{src.lower()}"
        if key not in self.map:
            self.map[key] = self._pick(kind, src, pool)
        return _case(src, self.map[key])

    def _digits(self, s: str) -> str:
        key = f"id:{s}"
        if key not in self.map:
            # The gate's digit-for-digit swap, keeping the leading zeros and a nonzero digit after them,
            # so the line's magnitude tokens (data.number_tokens, a model input) stay as they were.
            zeros = len(s) - len(s.lstrip("0"))
            for _ in range(8):
                out = "0" * zeros
                if zeros < len(s):
                    out += str(self.rng.randrange(1, 10)) + "".join(
                        str(self.rng.randrange(10)) for _ in range(len(s) - zeros - 1))
                if out != s:
                    break
            self.map[key] = out
        return self.map[key]

    def _email(self, m: re.Match) -> str:
        local, dom, tld = m.group(1), m.group(2), m.group(3)
        if _own(dom, self.own):
            return m.group(0)
        new_local = self._get("first", local, [f.lower() for f in TRAIN_FIRST])
        return f"{new_local}@{self.brand_map.get(dom.lower(), dom).lower()}{tld}"

    def _cell(self, m: re.Match) -> str:
        cell = m.group(2)
        low = cell.lower()
        if low in self.dict or low in CITY_SET or _own(cell, self.own):
            return m.group(0)
        return m.group(1) + self._get("place", cell, TRAIN_CITIES).upper()

    def _first(self, m: re.Match) -> str:
        w = m.group(0)
        if w in self.names and w.lower() not in self.dict and not _own(w, self.own):
            return self._get("first", w, TRAIN_FIRST)
        return w

    def __call__(self, text: str) -> str:
        if not text:
            return text
        t = EMAIL.sub(self._email, text)
        if self.brand_re is not None:
            t = self.brand_re.sub(lambda m: _case(m.group(0), self.brand_map[m.group(0).lower()])
                                  if "@" not in t[max(0, m.start() - 1):m.start()] else m.group(0), t)
        t = CITY_RE.sub(lambda m: self._get("city", m.group(0), TRAIN_CITIES), t)
        t = ZIP.sub(lambda m: f"{m.group(1)}{m.group(2)}{self._digits(m.group(3))}{m.group(4) or ''}", t)
        t = STATE.sub(lambda m: self._get("state", m.group(0), TRAIN_STATES), t)
        t = CELL_CAPS.sub(self._cell, t)
        t = TITLE.sub(self._first, t)
        # A surname already in the map moves wherever it stands ("Escalera, Jorge" as "Jorge Escalera"):
        # swap_texts reads the part twice, so the map is complete on the pass that counts.
        lasts = {k[5:]: v for k, v in self.map.items() if k.startswith("last:")}
        if lasts:
            t = re.sub(r"\b(" + "|".join(re.escape(k) for k in sorted(lasts, key=len, reverse=True)) + r")\b",
                       lambda m: _case(m.group(0), lasts[m.group(0).lower()]) if m.group(0)[:1].isupper()
                       else m.group(0), t, flags=re.I)
        firsts = sorted({v for k, v in self.map.items() if k.startswith("first:")}, key=len, reverse=True)
        if firsts:
            t = re.sub(r"\b(" + "|".join(map(re.escape, firsts)) + r")\s+([A-Z][a-z][a-zA-Z'-]+)\b",
                       lambda m: f"{m.group(1)} {self._get('last', m.group(2), TRAIN_LAST)}"
                       if m.group(2).lower() not in self.dict and not _own(m.group(2), self.own)
                       and m.group(2) not in TRAIN_FIRST and m.group(2) not in TRAIN_LAST else m.group(0), t)
        t = ID_TOKEN.sub(lambda m: m.group(0) if _own(m.group(1), self.own) or m.group(1) in NOT_ID_PREFIX
                         else f"{m.group(1)}{m.group(2)}{self._digits(m.group(3))}", t)
        return ID_AFTER.sub(lambda m: f"{m.group(1)}{m.group(2)}{self._digits(m.group(3))}", t)


def swap_texts(texts: list[str], company: str | None, rng: random.Random) -> list[str]:
    """The part's texts with every entity swapped consistently (one map for the whole part). The first
    read fills the map; the second applies it, so an entity reads the same on every line of the part."""
    fn = EntitySwap(texts, company, rng)
    for t in texts:
        if isinstance(t, str):
            fn(t)
    return [fn(t) if isinstance(t, str) else t for t in texts]


def build_words(dirs: list[str]) -> dict:
    """The detection lexicon from training files only: proper names that are not dictionary words (the gate's
    FIRST_SET test) and the dictionary words that occur capitalised in the training texts (the only ones the
    TITLE / surname / all-caps cell checks can meet). Holdouts and the selection deal are skipped."""
    import glob
    import os

    from .holdout import FIXED_HOLDOUT, SELECTION

    with open("/usr/share/dict/words") as f:
        dictionary = {w.strip().lower() for w in f if w.strip()[:1].islower()}
    with open("/usr/share/dict/propernames") as f:
        names = sorted({w.strip() for w in f if w.strip().isalpha() and len(w.strip()) >= 3
                        and w.strip().lower() not in dictionary})
    seen, files = set(), 0
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            if os.path.basename(path)[:6] in FIXED_HOLDOUT | SELECTION:
                continue
            files += 1
            for a in json.load(open(path)).get("atoms") or []:
                for m in re.finditer(r"\b[A-Z][A-Za-z'-]*\b", a.get("text") or ""):
                    if m.group(0).lower() in dictionary:
                        seen.add(m.group(0).lower())
    return {"_comment": "entity_swap detection lexicon (ml/c3/entity_swap.py build_words); detection only, "
                        f"never a replacement. {files} training files, holdouts and selection excluded.",
            "names": names, "dict_words": sorted(seen)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build-words", nargs="+", metavar="DIR", required=True)
    args = ap.parse_args()
    WORDS_FILE.write_text(json.dumps(build_words(args.build_words), indent=0) + "\n")
    print(f"wrote {WORDS_FILE}")
