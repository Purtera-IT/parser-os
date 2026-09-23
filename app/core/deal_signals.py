"""Sentences that are not scope still carry facts. Pull the fact out first.

"Here are the details for the small job I was discussing earlier" is not a
line item, and typing it as one is wrong -- but it says the job is SMALL, and
that it was discussed before this thread. "Diagram:" plus a link says a
drawing exists that the deal does not have. "It could lead to many more of the
same opportunity" says this account has room to grow.

Hiding those as small talk throws the fact away, which is worse than filing it
under the wrong type. So: read the signal, mint it as its own atom, and only
what is left over -- the thanks and the handoffs -- is small talk.

Live 010289 (Huzzard / CDW): three sentences, three facts, all previously lost.
"""
from __future__ import annotations

import re
from typing import Any

from app.core.ids import stable_id

#: How big the sender thinks this is. Deal Kit asks the question; the customer
#: usually answers it in passing, in the first line of the first mail.
_SCALE_RE = re.compile(
    r"\b(?P<small>small|quick|simple|little|one[- ]off|single[- ]door|tiny)\s+"
    r"(?:job|project|install(?:ation)?|ask|one|opportunity|deployment)\b"
    r"|\b(?P<large>large|big|major|full|enterprise[- ]wide|multi[- ]site|nationwide)\s+"
    r"(?:job|project|rollout|roll[- ]out|install(?:ation)?|deployment|program)\b",
    re.I,
)

#: A drawing, a floorplan, a photo, a spec sheet the sender is pointing at.
_ARTIFACT_WORD_RE = re.compile(
    r"\b(?P<what>diagram|drawing|floor ?plan|schematic|layout|photo|picture|screenshot|"
    r"spec ?sheet|cut ?sheet|data ?sheet|quote|proposal|scope document)\b",
    re.I,
)

#: A promise with a condition in front of it. "IF you all would be able to
#: do something like this, I will get a conversation going with the club
#: owner" is not a promise he made -- it is a promise waiting on US, and that
#: is the deal's critical path. Typed `commitment` it reads as the opposite.
_CONDITIONAL_RE = re.compile(
    r"\bif\s+(?P<who>you all|you|y'all|your team|we|i|they|the (?:client|customer|owner|club|installer))\b"
    r"[^.]{0,140}?,\s*(?P<then>i|we)\s*(?:'ll|will|would|can|shall)\b",
    re.I,
)

#: A line that announces what follows: "Here are the details for the small
#: job". Everything under it is one ask, and the readings on it -- how big the
#: job is, whose job it is -- belong to those atoms, not to the whole thread.
_OPENS_BLOCK_RE = re.compile(
    r"^\s*(?:here (?:are|is|'s)|below (?:are|is)|attached (?:are|is)|these are|the following (?:are|is))\s+"
    r"(?P<what>(?:the\s+|our\s+|my\s+)?[a-z][^.]{0,70})",
    re.I,
)

#: Someone undertakes to do something. Deal Kit cares who owes what, and a
#: promise conditional on OUR answer is a task on us before it is one on them.
_PROMISE_RE = re.compile(
    r"(?P<who>\bi\b|\bwe\b)\s*(?:'ll|will|shall|am going to|are going to)\s+(?P<what>[a-z][^.,;]{4,80})",
    re.I,
)

#: A party who is not in this thread being brought into it: the club owner,
#: the landlord, the installer. Live 010288's decision maker appears exactly
#: once, in a sentence a pipeline-chatter rule had hidden.
_INTRODUCES_RE = re.compile(
    r"\b(?:conversation|call|intro(?:duction)?|loop(?:ing)? in|put you in touch|speak|talk)\b"
    r"[^.]{0,40}?\b(?:with|to)\s+(?:the\s+|their\s+|our\s+)?"
    r"(?P<party>club owner|owner|landlord|tenant|gm\b|general manager|facilities manager|facilities|"
    r"installer|electrician|architect|director|principal|president|decision maker|end user|client)",
    re.I,
)

#: Room to grow on this account: real, worth knowing, never scope.
_EXPANSION_RE = re.compile(
    r"\b(?:lead to (?:many )?more|more of the same|many more of the same|"
    r"(?:could|would|might) (?:lead|turn) into|roll ?out to (?:the )?(?:other|rest)|"
    r"(?:other|additional|more) (?:clubs|sites|locations|stores|branches))\b",
    re.I,
)

SIGNAL_FLAG = "carries_signal"


def _value(atom: Any) -> dict:
    v = getattr(atom, "value", None)
    return v if isinstance(v, dict) else {}


def _text(atom: Any) -> str:
    return str(getattr(atom, "raw_text", "") or "")


def _reads(atom: Any, key: str, value: Any, *, why: str, confidence: float) -> None:
    """Record what this atom TELLS US. Not a new atom: nobody said it.

    ``reads`` is the atom's meaning -- the thing Deal Kit reasons with ("a
    small job" changes how many PMs it plans for) and the thing a head is
    trained to produce. The rule that found it is scaffolding: it exists to
    put a first label in front of a human, and a head trained on those labels
    answers from meaning instead of from a pattern.
    """
    val = _value(atom)
    reads = list(val.get("reads") or [])
    if any(r.get("key") == key for r in reads):
        return
    reads.append({"key": key, "value": value, "why": why, "confidence": confidence, "source": "rule"})
    val["reads"] = reads
    atom.value = val
    flags = list(getattr(atom, "review_flags", None) or [])
    if SIGNAL_FLAG not in flags:
        flags.append(SIGNAL_FLAG)
    atom.review_flags = flags


def _mark(atom: Any, signal: str) -> None:
    """Say on the atom that a fact was read out of it, so the small-talk pass
    leaves it alone and the PM can see why it is still here."""
    val = _value(atom)
    signals = list(val.get("signals") or [])
    if signal not in signals:
        signals.append(signal)
    val["signals"] = signals
    atom.value = val
    flags = list(getattr(atom, "review_flags", None) or [])
    if SIGNAL_FLAG not in flags:
        flags.append(SIGNAL_FLAG)
    atom.review_flags = flags


def _new_atom(*, project_id: str, source: Any, atom_type_name: str, text: str, value: dict, confidence: float):
    from app.core.schemas import AtomType, AuthorityClass, EvidenceAtom, ReviewStatus

    return EvidenceAtom(
        id=stable_id("atm", project_id, str(getattr(source, "artifact_id", "")), value.get("kind", "signal"), text),
        project_id=project_id,
        artifact_id=str(getattr(source, "artifact_id", "") or ""),
        atom_type=getattr(AtomType, atom_type_name),
        raw_text=text,
        normalized_text=text.lower(),
        value={**value, "derived_from": str(getattr(source, "id", "") or "")},
        entity_keys=list(getattr(source, "entity_keys", None) or []),
        source_refs=list(getattr(source, "source_refs", None) or []),
        receipts=list(getattr(source, "receipts", None) or []),
        authority_class=AuthorityClass.machine_extractor,
        confidence=confidence,
        review_status=ReviewStatus.needs_review,
        review_flags=["derived_signal"],
        parser_version=str(getattr(source, "parser_version", "") or "signal"),
    )


#: A drawing is a picture or a PDF. The mail that MENTIONS the diagram is not
#: the diagram: live 010289's note is filed as "…-The Ask w diagram link.txt",
#: which read as "we have it" and cost us the one fact that mattered.
_ARTIFACT_FILE_RE = re.compile(r"\.(png|jpe?g|gif|webp|bmp|svg|pdf|dwg|vsdx?|ai|tiff?)$", re.I)


def _have_artifact(word: str, filenames: list[str]) -> bool:
    w = word.lower().replace(" ", "")
    for f in filenames:
        if not _ARTIFACT_FILE_RE.search(f.strip()):
            continue
        if w in f.lower().replace(" ", "").replace("_", "").replace("-", ""):
            return True
    return False


def _side_of(party: dict | None) -> str:
    """`us`, `partner` or `customer` for a party the parser stamped."""
    if not isinstance(party, dict):
        return ""
    if str(party.get("side") or "") == "ours":
        return "us"
    role = str(party.get("role") or "").lower()
    return "partner" if role in {"reseller", "vendor", "manufacturer", "installer"} else "customer"


def _blocked_on(value: dict, who: str) -> str:
    """Whose answer the condition waits on, read from who is speaking to whom.

    "If YOU all would be able to…" means us when a reseller writes it to us,
    and means them when we write it to them. The words alone cannot say which.
    """
    who = who.lower().strip()
    said_by = value.get("said_by") if isinstance(value.get("said_by"), dict) else None
    said_to = value.get("said_to") if isinstance(value.get("said_to"), list) else []
    speaker = _side_of(said_by)
    listener = _side_of(said_to[0] if said_to and isinstance(said_to[0], dict) else None)
    if who in {"you", "you all", "y'all", "your team"}:
        return listener or ("us" if speaker and speaker != "us" else "")
    if who in {"we", "i"}:
        return speaker
    return "customer"

def extract_deal_signals(atoms: list[Any], *, project_id: str, filenames: list[str] | None = None) -> list[Any]:
    """Read what a sentence TELLS US onto the sentence itself.

    Returns the few atoms that are genuinely new WORK (a drawing to go and
    get), never a restatement of something already said: "the sender calls
    this a small job" is not a statement anyone made, it is what their
    sentence means, and it now rides on that sentence as ``reads``.
    """
    made: list[Any] = []
    seen: set[str] = set()
    names = list(filenames or [])

    for atom in atoms:
        text = " ".join(_text(atom).split())
        if not text:
            continue
        # A wrapped link runs to 700 characters on its own; read the head of
        # the line, and quote only what a person would read.
        head = text[:400]
        quote = text[:300]
        val = _value(atom)

        m = _SCALE_RE.search(head)
        if m:
            scale = "small" if m.group("small") else "large"
            key = f"scale:{scale}"
            _mark(atom, key)
            _reads(atom, "job_scale", scale, why=m.group(0).lower(), confidence=0.72)

        # A picture the sender points at. If the deal does not hold a file by
        # that name, somebody has to go and get it before the job is scoped.
        am = _ARTIFACT_WORD_RE.search(head)
        if am and (val.get("image_url") or re.search(r"https?://|attach|linked|below|see the", head, re.I)):
            what = am.group("what").lower()
            key = f"artifact:{what}"
            _mark(atom, key)
            _reads(atom, "points_at_artifact", what, why=f"names a {what}", confidence=0.7)
            if key not in seen and not _have_artifact(what, names):
                seen.add(key)
                url = str(val.get("image_url") or val.get("url") or "")
                url = str(val.get("image_url") or val.get("url") or "")
                _reads(atom, "needs_artifact", what, why=f"names a {what} the deal does not hold",
                       confidence=0.7)
                _value(atom)["artifact_url"] = url or None

        pm = _PROMISE_RE.search(head)
        if pm:
            _mark(atom, "promise")
            _reads(atom, "commitment", " ".join(pm.group("what").split())[:80],
                   why=" ".join(pm.group(0).split())[:90], confidence=0.7)

        im = _INTRODUCES_RE.search(head)
        if im:
            _mark(atom, "introduces_party")
            _reads(atom, "introduces_party", im.group("party").lower(),
                   why="brings someone into the deal who is not in this thread", confidence=0.66)

        cm = _CONDITIONAL_RE.search(head)
        if cm:
            side = _blocked_on(val, cm.group("who"))
            if side:
                _reads(atom, "blocked_on", side,
                       why=f'conditional: "if {cm.group("who").lower()} …, {cm.group("then").lower()} will …"',
                       confidence=0.62)

        om = _OPENS_BLOCK_RE.match(text.strip())
        if om:
            _reads(atom, "opens_block", om.group("what").strip().lower().rstrip(" ,;:"),
                   why="the line announces what follows; the atoms under it are its detail",
                   confidence=0.6)

        xm = _EXPANSION_RE.search(head)
        if xm:
            _mark(atom, "expansion")
            # WHAT would repeat, in the sender's words. `True` taught a head the
            # weaker half of its own question, while `commitment` beside it has
            # always carried what was promised.
            # The phrase, finished: "lead to many more" on its own drops the
            # thing that repeats, which is the half worth having.
            said = head[xm.start():].split(".")[0].split(";")[0].strip().lower()[:90]
            _reads(atom, "expansion", said or xm.group(0).strip().lower(),
                   why="the sender says this could repeat across more of their sites",
                   confidence=0.65)

    return made


__all__ = ["extract_deal_signals", "SIGNAL_FLAG"]
