"""A layer name is a sentence in a standard, not a label to memorise.

The first version of the drawing derivation matched layers by literal name --
``AR-WALL-N`` for new partition, ``AR-WALL-DEM-N`` for demolition. That works
on exactly one architect's files. BR Design wrote ``AR-``; the US National CAD
Standard says ``A-``; other firms ship ``A-WALL-NEWW`` or their own scheme
entirely. A hardcoded list quietly finds nothing on the next deal's drawing and
reports no new partition rather than reporting that it could not tell.

The names vary. The GRAMMAR does not:

    [Discipline] - [Major] - [Minor...] - [Status]
         A          WALL        DEM          N

Of SP-6's 71 layers, 38 parse cleanly this way, and the ones that do not are
mostly CAD furniture (``DEFPOINTS``, ``DIMENSIONS``, ``0``) plus one
firm-specific discipline, ``BC-`` for base building.

So this module reads the grammar and answers by ROLE -- "is this a new-work
wall layer?" -- which holds across firms, instead of by name, which does not.
Where a file departs from the standard the answer is `None`, and a caller that
gets `None` should say it could not tell rather than say zero.

Reference: NCS / AIA CAD Layer Guidelines, discipline + major group codes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: NCS discipline designators. One or two characters; the second is a
#: firm-level qualifier, which is why ``AR`` still reads as Architectural.
DISCIPLINES = {
    "A": "architectural", "I": "interiors", "E": "electrical",
    "T": "telecommunications", "M": "mechanical", "P": "plumbing",
    "S": "structural", "F": "fire_protection", "C": "civil",
    "L": "landscape", "Q": "equipment", "G": "general", "X": "other",
}

#: The disciplines that answer a cabling question. Their ABSENCE is a finding:
#: a sheet with no T or E layer cannot give tray runs or outlet positions.
CABLING_DISCIPLINES = ("T", "E")

#: NCS major group codes this module reasons about, mapped to a role. The
#: four-character code is the standard; the alternatives are what firms
#: actually type.
MAJOR_ROLES = {
    "WALL": "partition", "PRHT": "partition", "PART": "partition",
    "DOOR": "door", "GLAZ": "glazing", "FLOR": "floor",
    "CLNG": "ceiling", "RCP": "ceiling",
    "COMM": "telecom", "DATA": "telecom", "TELE": "telecom",
    "POWR": "power", "LITE": "lighting",
    "FURN": "furniture", "EQPM": "equipment",
}

#: Status suffixes. ``N`` new, ``E`` existing, ``D`` demolish, ``F`` future,
#: ``X`` not in contract. Firms also spell them out.
STATUS = {
    "N": "new", "NEW": "new", "NEWW": "new",
    "E": "existing", "EXST": "existing", "EXIST": "existing", "EXSTG": "existing",
    "D": "demolish", "DEM": "demolish", "DEMO": "demolish",
    "F": "future", "FUTR": "future", "FUTURE": "future",
    "X": "not_in_contract", "NIC": "not_in_contract",
}

#: CAD's own furniture -- never content.
NON_CONTENT = {"0", "DEFPOINTS", "DIMENSIONS", "TITLE", "TITLEBLOCK", "VIEWPORT"}

_PART = re.compile(r"[-_ ]+")


@dataclass(frozen=True)
class Layer:
    raw: str
    discipline: str | None      #: single-letter NCS code, e.g. "A"
    major: str | None           #: NCS major group, e.g. "WALL"
    role: str | None            #: what the major group means, e.g. "partition"
    status: str | None          #: "new" | "existing" | "demolish" | "future" | ...
    demolition: bool = False    #: a DEM token in the MINOR field, not the status

    @property
    def is_content(self) -> bool:
        return self.raw.upper() not in NON_CONTENT

    @property
    def intent(self) -> str | None:
        """What the layer is FOR, which is not always its status field.

        ``AR-WALL-DEM-N`` is major WALL, minor DEM, status N -- read strictly
        that is "new", and reading it strictly reported zero demolition on a
        sheet with 144 feet of it. The minor field is where demolition is
        said; the status then qualifies the demolition, not the wall.
        """
        if self.demolition:
            return "future_demolition" if self.status == "future" else "demolish"
        return self.status


def parse_layer(name: str) -> Layer:
    """Read a layer name as discipline / major / status.

    Deliberately forgiving in one direction only: an unrecognised part becomes
    ``None`` rather than a guess, because a wrong role is worse than no role --
    it turns "I cannot tell how much new wall there is" into a confident zero.
    """
    raw = (name or "").strip()
    parts = [p for p in _PART.split(raw.upper()) if p]
    if not parts:
        return Layer(raw, None, None, None, None)

    head = parts[0]
    # "AR" is Architectural with a firm qualifier; "BC" (base building) is not
    # a standard code at all, and reads as unknown rather than as "C".
    disc = head[0] if head[:1] in DISCIPLINES and len(head) <= 2 else None
    if disc and len(head) == 2 and head not in ("AR", "AI", "AE", "AS", "AD", "AC"):
        # A two-letter head whose second letter is not a known qualifier is
        # more likely a firm's own code (BC, DM, CP) than a discipline.
        disc = head[0] if head[0] in DISCIPLINES and head[1] in "RIESDC" else None

    major = role = None
    for p in parts[1:]:
        if p in MAJOR_ROLES:
            major, role = p, MAJOR_ROLES[p]
            break

    # Status is a trailing token, and NCS puts it last. Scanned from the end so
    # "AR-WALL-DEM-N" reads new-work demolition drafting as NEW, matching what
    # the layer list on SP-6 actually means.
    status = None
    for p in reversed(parts[1:]):
        if p in STATUS:
            status = STATUS[p]
            break

    # Demolition is said in the MINOR field and qualified by the status, so it
    # is tracked separately -- see `Layer.intent`.
    mid = parts[1:]
    if major in mid:
        mid = mid[mid.index(major) + 1:]
    demolition = any(p in ("DEM", "DEMO", "DEMOLISH") for p in mid)

    return Layer(raw, disc, major, role, status, demolition)


def layers_with_role(names: list[str], role: str, status: str | None = None) -> list[str]:
    """Every layer playing this role, whatever the firm calls it."""
    out = []
    for n in names:
        lay = parse_layer(n)
        if lay.role != role or not lay.is_content:
            continue
        # Matched on INTENT, not the raw status field: a demolition layer is
        # demolition whatever its trailing status says.
        if status is not None and lay.intent != status:
            continue
        out.append(n)
    return out


#: How much of a drawing has to parse before its SILENCE means anything.
#:
#: One accidental match is not evidence of a standard. A real conveyor drawing
#: in this corpus has 75 layers named `$AUDIT-BAD-LAYER`, `-0`, `001`, `02J`
#: and `0_...`, of which exactly two parsed as a discipline by coincidence --
#: enough, under the first version of this check, to certify the file as
#: standards-following and then report "no telecom or electrical information"
#: about a MOTOR CONTROL PANEL drawing. Both findings were confident and both
#: were false.
STANDARD_FRACTION = 0.25


def follows_standard(names: list[str]) -> bool:
    """Does this drawing speak the layer standard well enough to be read?

    Requires a quarter of its content layers to carry BOTH a discipline and a
    recognised major group. Two stray matches out of seventy-five do not make
    a drawing legible, and treating them as if they did turns "I cannot read
    this file" into a confident statement about what the file omits.
    """
    content = [parse_layer(n) for n in names]
    content = [p for p in content if p.is_content and p.raw.strip()]
    if len(content) < 4:
        return False
    legible = sum(1 for p in content if p.discipline and p.role)
    return legible / len(content) >= STANDARD_FRACTION


def has_discipline(names: list[str], codes: tuple[str, ...]) -> bool | None:
    """Is any of these disciplines on the sheet?

    ``None`` when the drawing does not follow the standard closely enough for
    its silence to mean anything -- an unreadable file omits nothing, it just
    cannot be read, and those are different answers.
    """
    if not follows_standard(names):
        return None
    return any(parse_layer(n).discipline in codes for n in names)
