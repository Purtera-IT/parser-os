"""Where the deal IS, as a handful of lines instead of forty atoms.

A deal's mailbox carries two different kinds of sentence, and only one of them
belongs in a deal kit.

The first states what we will build, price and commit to: 106 workstations,
two Cat6A drops each, six 48-port patch panels. That is content, and every
sentence of it earns an atom.

The second is the deal MOVING. Who is doing the site survey, which week, is
the seller free Tuesday, did anyone tell Rich, thanks-got-the-floorplans. On
live 010180 that was 43 of 252 non-rejected atoms -- a sixth of the deal --
and not one line of it will ever appear in a SOW. As individual atoms they do
not merely waste space: they compete for a PM's attention with the two numbers
on the deal that move money.

So they are consolidated. Six lines say what forty atoms said:

    stage         awaiting site survey
    next step     onsite walkthrough, 7 Penn Plaza
    survey owner  PurTera ops
    survey terms  free if we win; invoiced if we lose
    price basis   ROM -- NOT survey-backed
    blocks        firm SOW / deal kit until the survey completes

The last two are the point. 010180's $110,108 was put together by PurTera's
own PM, working from the customer's floor plan and the email thread -- which
is exactly why it is careful and explicitly basic. Nobody had been on site.
The deal then says so itself: "has already provided a budgetary ROM ... and is
now being asked to perform a site survey/walkthrough to generate a firm
quote."

That is not a criticism of the estimate. It is the estimate's OWN statement of
what it is, and a deal kit built on it would turn a number its author called
budgetary into a firm promise. The gate exists to keep the PM's caveat
attached to the number as it travels.

Everything here is DERIVED, so like every other derived claim it carries what
it assumed, and it says nothing when the evidence is not there.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: A price stated as rough, by the people who produced it.
_ROM = re.compile(r"\bROM\b|budgetary (?:number|estimate|numbers|pricing)|ballpark"
                  r"|rough order of magnitude", re.I)

#: The deal asking for the survey that would replace it. A SURVEY, asked for.
#: "Walk-through" and "site visit" alone are job steps, not a pending survey:
#: live 000132 read "On-site visit once per week", "one coordinated site
#: visit" and "complete an onsite walk-through before beginning the removal"
#: as the deal waiting on a survey, and staged it "awaiting site survey".
_SURVEY_WORD = re.compile(r"\bsurvey(?:s|ed|ing)?\b", re.I)

#: Someone asking for it, planning it or waiting on it.
_SURVEY_REQUEST = re.compile(
    r"\b(?:wants?|wanted|would like|ask(?:s|ed|ing)?|request(?:s|ed|ing)?"
    r"|needs?|needed|requires?|required|schedul\w*|set up|book|arrange"
    r"|perform|conduct|do an?|can you|could you|please|availability"
    r"|dial in|generate|firm quote|true quote|before we can|await\w*"
    r"|wait(?:ing)? (?:on|for)|pending|once|until|after)\b", re.I)

#: A recurring visit is a service term, never a pending survey.
_RECURRING = re.compile(r"\b(?:once|twice|\d+\s*(?:x|times)) (?:a|per|every)\b|\bper (?:week|month)\b"
                        r"|\b(?:weekly|monthly)\b", re.I)


def _asks_for_a_survey(text: str) -> bool:
    """Is a survey being asked for (or waited on) here, rather than merely
    named as a step of the work?"""
    if not _SURVEY_WORD.search(text) or _RECURRING.search(text):
        return False
    return bool(_SURVEY_REQUEST.search(text))

#: The survey having happened. Deliberately narrow: a survey is done when
#: somebody says it is, not when one was merely scheduled.
_SURVEY_DONE = re.compile(
    r"survey (?:is |was |has been )?(?:complete|completed|done|finished)"
    r"|completed the (?:site )?(?:survey|walk ?through)"
    r"|after the (?:site )?survey was (?:done|completed)", re.I)

#: A completion phrase can be a PLAN rather than a report, and on 010180 both
#: of them were:
#:
#:     "we can tackle the cabling ... once survey is completed"
#:     "wait until the wireless survey is complete so that we know where APs go"
#:
#: Read as completions they said the price had been checked against the site
#: when nobody from PurTera had been there -- which would switch OFF the one
#: gate this module exists to hold, and do it silently. A future survey is the
#: reason the deal is waiting, not evidence that it is not.
_CONDITIONAL = re.compile(
    r"\b(once|after|when|until|til|till|if|pending|before|following|"
    r"prior to|awaiting|wait(?:ing)? for|subject to)\b", re.I)

#: How far back to look for that conditional. A clause, not a paragraph.
_CONDITIONAL_WINDOW = 40


def _reports_a_completed_survey(text: str) -> bool:
    """Does this sentence REPORT a survey, rather than plan around one?"""
    for m in _SURVEY_DONE.finditer(text):
        lead = text[max(0, m.start() - _CONDITIONAL_WINDOW):m.start()]
        if not _CONDITIONAL.search(lead):
            return True
    return False

#: A firm number being asked for. This is what a ROM is NOT.
_WANTS_FIRM = re.compile(
    r"firm (?:quote|price|pricing|number)|true quote|detailed SOW|dial in"
    r"|final (?:quote|pricing)", re.I)

#: Pure coordination: no deal kit will ever contain it.
_COORDINATION = (
    ("scheduling", re.compile(
        r"provide availability|what days|specific days|week of the|next week"
        r"|are you free|calendar invite|does that work for you", re.I)),
    ("acknowledgement", re.compile(
        r"^(thanks|thank you|gotcha|sounds good|will do|got it|perfect|noted"
        r"|yes,? got|ok[.,!]?$)", re.I)),
    ("chasing", re.compile(
        r"following up|circling back|checking in|any update|per my last"
        r"|let me know your thoughts|bumping this", re.I)),
)


@dataclass
class StateLine:
    key: str
    value: str
    why: str
    assumption: str = ""
    evidence: list[str] = field(default_factory=list)
    #: The atoms the evidence came from, so the derived line is pinned to the
    #: artifact that said it rather than to whichever atom happened to be first.
    evidence_atoms: list[Any] = field(default_factory=list, repr=False)


@dataclass
class DealState:
    lines: list[StateLine] = field(default_factory=list)
    coordination: list[Any] = field(default_factory=list)

    def get(self, key: str) -> StateLine | None:
        return next((l for l in self.lines if l.key == key), None)


def _text(atom: Any) -> str:
    return " ".join(str(getattr(atom, "raw_text", "") or "").split())


def _money(text: str) -> str:
    m = re.search(r"\$\s?([\d,]+(?:\.\d{2})?)\s?[kK]?", text)
    return m.group(0).strip() if m else ""


def read_deal_state(atoms: list[Any]) -> DealState:
    """What the deal's own sentences say about where it stands.

    Returns only what the evidence supports. A deal with no ROM and no survey
    talk produces no lines at all, which is correct: most deals are not
    waiting on anything.
    """
    state = DealState()
    rom: list[Any] = []
    wants_survey: list[Any] = []
    survey_done: list[Any] = []
    wants_firm: list[Any] = []

    for atom in atoms:
        t = _text(atom)
        if not t:
            continue
        if _ROM.search(t):
            rom.append(atom)
        if _asks_for_a_survey(t) and not _reports_a_completed_survey(t):
            wants_survey.append(atom)
        if _reports_a_completed_survey(t):
            survey_done.append(atom)
        if _WANTS_FIRM.search(t):
            wants_firm.append(atom)
        for kind, rx in _COORDINATION:
            if rx.search(t):
                state.coordination.append(atom)
                break

    # ---- the price basis, and what follows from it -------------------------
    if rom:
        amount = next((_money(_text(a)) for a in rom if _money(_text(a))), "")
        priced = f" ({amount})" if amount else ""
        surveyed = bool(survey_done)
        state.lines.append(StateLine(
            key="price_basis",
            value=("ROM, survey-backed" if surveyed
                   else f"ROM{priced} — NOT survey-backed"),
            why=("The number on this deal is described by the people who "
                 "produced it as budgetary. " +
                 ("A completed survey is on the record, so it has been "
                  "checked against the site."
                  if surveyed else
                  "No completed survey is on the record, so nothing has "
                  "checked it against the site.")),
            assumption="a price called budgetary or a ROM is a rough number, "
                       "and a survey is complete only when somebody says so",
            evidence=[_text(a)[:120] for a in rom[:3]],
            evidence_atoms=list(rom[:3])))

        if not surveyed and (wants_survey or wants_firm):
            state.lines.append(StateLine(
                key="blocks",
                value="firm SOW / deal kit — until the survey completes",
                why=("A deal kit is a firm promise. Building one on a ROM that "
                     "no survey has checked makes a firm promise out of a rough "
                     "guess, and the deal itself says the customer is asking for "
                     "a firm quote that the survey is supposed to produce."),
                assumption="a deal kit should not be generated from an "
                           "unsurveyed budgetary number",
                evidence=[_text(a)[:120] for a in (wants_firm or wants_survey)[:3]],
                evidence_atoms=list((wants_firm or wants_survey)[:3])))

    # ---- what happens next -------------------------------------------------
    if wants_survey and not survey_done:
        state.lines.append(StateLine(
            key="next_step",
            value="site survey / onsite walkthrough",
            why="The survey is the step the deal is waiting on: it is what "
                "turns the ROM into a firm number.",
            assumption="a survey asked for and not reported complete is "
                       "outstanding",
            evidence=[_text(a)[:120] for a in wants_survey[:3]],
            evidence_atoms=list(wants_survey[:3])))
        state.lines.append(StateLine(
            key="stage",
            value="awaiting site survey",
            why="Derived from the step outstanding.",
            assumption="the outstanding step is the stage",
            evidence=[_text(a)[:120] for a in wants_survey[:3]],
            evidence_atoms=list(wants_survey[:3])))

    return state
