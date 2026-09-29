"""What a drawing states once you MEASURE it.

A room tag on its own is worth nothing to a deal. "PANTRY" names a room the PM
already knew about; it decides nothing, prices nothing and cannot be wrong.
The reason to read CAD rather than a picture of it is that every tag carries a
coordinate and the sheet carries a scale -- and those two together answer the
question a bare tag cannot: HOW FAR.

On a structured-cabling job that question has a hard answer. TIA-568 allows a
90 m permanent link for a horizontal Cat6A run. Exceed it and the drop does not
merely underperform, it fails certification, and the fix is a second closet --
tens of thousands of dollars discovered after the quote is signed. Nobody on
this deal had measured it, because until the DWG parsed there was nothing to
measure.

This module does the measuring. It is deliberately conservative: it reports
what it assumed, it reports the raw distance beside the derated one, and when
the headroom is too small for tag positions to settle the question it says so
instead of guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.parsers.cad_layers import (
    CABLING_DISCIPLINES,
    follows_standard,
    has_discipline,
    layers_with_role,
)
from typing import Any

#: TIA-568 horizontal permanent link, in feet. The 100 m channel includes patch
#: cords at both ends; the 90 m is what the installed cable may be.
CAT6A_PERMANENT_LINK_FT = 295.0

#: Cable does not fly. It runs orthogonally in tray, so a plan distance is the
#: MANHATTAN distance between two points, not the straight line.
#:
#: These two allowances are ordinary estimating practice, named here so a
#: reviewer can argue with them rather than having to discover them:
ROUTING_SLACK = 1.15   #: detours around cores, ducts and demising walls
RISER_ALLOWANCE_FT = 24.0  #: up into the plenum and back down, both ends

#: $INSUNITS. A drawing that does not say is not guessed at.
_UNITS_TO_FEET = {1: 1 / 12.0, 2: 1.0, 4: 1 / 304.8, 5: 1 / 30.48, 6: 3.28084}

_CLOSET = re.compile(r"^(IT|MDF|IDF|TELECOM|TELE|COMM|DATA)\b|(IT|SERVER|TELECOM|COMM)\s*(CLOSET|ROOM|RM)", re.I)


@dataclass
class Tag:
    text: str
    x: float
    y: float
    layer: str


@dataclass
class Reach:
    """The furthest drop the drawing implies, and whether it fits."""
    closet: str
    furthest: str
    straight_ft: float
    routed_ft: float
    permanent_link_ft: float
    headroom_ft: float
    verdict: str                     #: "clear" | "tight" | "over" | "unknown"
    note: str
    ranked: list[tuple[str, float]] = field(default_factory=list)


def tags_from(doc: Any, layer: str = "ROOM-TAG") -> list[Tag]:
    """Every room tag with a position, across every layout of the sheet."""
    out: list[Tag] = []
    spaces = [doc.modelspace()]
    try:
        spaces += [lay for lay in doc.layouts if getattr(lay, "name", "") != "Model"]
    except Exception:  # noqa: BLE001 - a layout we cannot enumerate is not fatal
        pass
    seen: set[tuple] = set()
    for sp in spaces:
        for e in sp:
            if e.dxftype() not in ("TEXT", "MTEXT"):
                continue
            if layer and e.dxf.layer != layer:
                continue
            try:
                text = " ".join((e.plain_text() if e.dxftype() == "MTEXT"
                                 else e.dxf.text).split()).strip()
                p = e.dxf.insert
                key = (text.upper(), round(float(p.x), 1), round(float(p.y), 1))
            except Exception:  # noqa: BLE001
                continue
            if not text or key in seen:
                continue
            seen.add(key)
            out.append(Tag(text, float(p.x), float(p.y), e.dxf.layer))
    return out


def feet_per_unit(doc: Any) -> float | None:
    """The drawing's own declared units, or None. Never assumed."""
    try:
        return _UNITS_TO_FEET.get(int(doc.header.get("$INSUNITS", 0)))
    except Exception:  # noqa: BLE001
        return None


#: Two points further apart than this are not in the same drawing. A tenant
#: floor is on the order of 100-200 ft across; CAD model space routinely holds
#: several plans side by side, hundreds of feet apart.
SAME_PLAN_FT = 400.0


def one_plan(points: list[Tag], origin: Tag, ft_per_unit: float,
             reach_ft: float = SAME_PLAN_FT) -> list[Tag]:
    """Only the points belonging to the SAME drawing as ``origin``.

    Model space is not one plan. SP-6 stacks a legend, a test-fit key, the
    tenant plan and a 106-desk furniture plan in one model space, hundreds of
    feet apart -- and measuring from the tenant plan's IT closet to the
    furniture plan's desks produced a 740 ft run and a confident verdict that
    the floor could not be cabled. It measured between two different drawings.

    Grown outward from the origin so an L-shaped floor stays whole, rather
    than taken as a fixed box around it.
    """
    taken = [origin]
    pool = [p for p in points if p is not origin]
    changed = True
    while changed:
        changed = False
        for p in list(pool):
            if any(abs(p.x - q.x) + abs(p.y - q.y) <= reach_ft / ft_per_unit
                   for q in taken):
                taken.append(p); pool.remove(p); changed = True
    return taken


#: There is deliberately no module-level place to stash the units.
#:
#: There was: a one-element list, set by `cable_reach` before it called
#: `one_plan`. It reads as a harmless trick and it is a correctness bug the
#: moment two drawings are parsed at once -- the second overwrites the first's
#: scale mid-measurement, and the distances come out wrong SILENTLY, which is
#: the worst way for a measurement to be wrong. The units are a property of
#: one drawing, so they travel as an argument.


def cable_reach(tags: list[Tag], ft_per_unit: float | None) -> Reach | None:
    """The worst horizontal run from the telecom closet, and whether it fits.

    Returns None when the drawing cannot answer: no closet tag, no declared
    units, or nothing to measure to. A finding we cannot stand behind is worse
    than no finding, because it will be believed.
    """
    if not tags or not ft_per_unit:
        return None
    closet = next((t for t in tags if _CLOSET.search(t.text)), None)
    if closet is None:
        return None

    # Measure inside ONE drawing. See `one_plan`.
    tags = one_plan(tags, closet, ft_per_unit)

    ranked: list[tuple[str, float, float]] = []
    for t in tags:
        if t is closet:
            continue
        dx, dy = abs(t.x - closet.x), abs(t.y - closet.y)
        ranked.append((t.text, (dx + dy) * ft_per_unit,
                       ((dx * dx + dy * dy) ** 0.5) * ft_per_unit))
    if not ranked:
        return None
    ranked.sort(key=lambda r: -r[1])
    name, routed, straight = ranked[0]

    link = routed * ROUTING_SLACK + RISER_ALLOWANCE_FT
    headroom = CAT6A_PERMANENT_LINK_FT - link

    # A tag sits where its TEXT is placed, not at the furthest outlet in the
    # room, so `routed` is a floor on the real run. That is only conclusive
    # when the headroom can absorb the difference -- hence three verdicts and
    # not a bare pass/fail.
    if headroom < 0:
        verdict = "over"
        note = (f"The furthest room on this plan ({name}) is {routed:.0f}' of routed cable "
                f"from {closet.text}, which derates to a {link:.0f}' permanent link -- past "
                f"the {CAT6A_PERMANENT_LINK_FT:.0f}' Cat6A limit by {-headroom:.0f}'. Drops at "
                f"that end will not certify from this closet. Either a second closet is in "
                f"scope or the cable path has to be shortened, and neither is priced.")
    elif headroom < routed * 0.5:
        verdict = "tight"
        note = (f"The furthest room ({name}) works out at a {link:.0f}' permanent link against "
                f"the {CAT6A_PERMANENT_LINK_FT:.0f}' Cat6A limit -- only {headroom:.0f}' spare. "
                f"A room tag sits where its text is placed, not at the furthest outlet in the "
                f"room, so the real run is longer than this by an unknown margin, and {headroom:.0f}' "
                f"is not enough to absorb it. This needs measuring on the site walk before "
                f"anyone commits to a single closet.")
    else:
        verdict = "clear"
        note = (f"Furthest room on the plan is {name}, {routed:.0f}' of routed cable from "
                f"{closet.text} ({straight:.0f}' straight). With {round((ROUTING_SLACK-1)*100)}% "
                f"routing slack and a {RISER_ALLOWANCE_FT:.0f}' riser allowance that is a "
                f"{link:.0f}' permanent link against the {CAT6A_PERMANENT_LINK_FT:.0f}' Cat6A "
                f"limit -- {headroom:.0f}' of headroom. A tag marks where its text sits rather "
                f"than the furthest outlet, so the true run is longer, but not by {headroom:.0f}'. "
                f"One closet reaches this floor.")
    return Reach(closet.text, name, straight, routed, link, headroom, verdict, note,
                 [(n, r) for n, r, _ in ranked[:8]])


def tag_census(tags: list[Tag]) -> dict[str, int]:
    """How many times the plan tags each room name -- the count to check a
    schedule row against. A schedule that says 2 and a plan that tags 1 is a
    disagreement worth a PM's eye; agreement is worth recording as agreement."""
    census: dict[str, int] = {}
    for t in tags:
        census[t.text.upper()] = census.get(t.text.upper(), 0) + 1
    return census


# ============================ reading the sheet, not listing it =============
#
# A room tag is not a finding. "PANTRY" names a room the PM already knew about,
# and twenty-three of them are twenty-three rows of noise that bury the two
# things on the sheet that actually decide money.
#
# Everything below is derived rather than read: counted, measured, or noticed
# to be ABSENT. Each finding carries the assumption it rests on, so a reviewer
# can argue with the assumption instead of reverse-engineering it, and so a
# head has something stable to learn over.

#: Furniture and fittings worth counting, and what a count of them decides.
_COUNTS = (
    (r"\bDESK\b|WORKSTATION|WORKSTN", "workstation",
     "Each desk drawn is a position that needs drops. A drawn count is "
     "independent of whatever a call summary said the headcount was."),
    (r"TV\b|MONITOR|DISPLAY|FLAT ?SCREEN|PROJECT(OR|ION)", "display",
     "A display drawn on the plan is an AV position, and AV drops are usually "
     "priced from a room count rather than a device count."),
    (r"DOOR", "door",
     "Doors drive card-reader positions and cable penetrations."),
)

#: Wall linework worth measuring, addressed by ROLE and INTENT rather than by
#: layer name. The first version hardcoded "AR-WALL-N", which is one firm's
#: spelling of the standard's "A-WALL-N" -- it would have found nothing on the
#: next architect's drawing and reported no new partition rather than saying it
#: could not tell. See `app.parsers.cad_layers`.
_LENGTHS = (
    (("partition", "new"), "new_partition",
     "New partition. Linear feet of new wall is how much construction is "
     "actually happening, and every new wall is a potential penetration."),
    (("partition", "demolish"), "demolished_partition",
     "Wall being demolished -- work that has to happen before cable pulls."),
    (("partition", "future_demolition"), "future_partition",
     "Wall marked future. Phasing nobody on this deal has mentioned."),
)

#: A discipline that is NOT on the sheet. Its absence is the finding: it says
#: what this drawing cannot answer, which is otherwise discovered late.
_DISCIPLINES = (
    (r"TRAY|CONDUIT|TELE|DATA|COMM|ELEC|OUTLET|DEVICE|POWER", "telecom_electrical",
     "No telecom or electrical layer, so this sheet cannot give tray runs, "
     "outlet positions or home-run paths. It is an architectural test-fit. "
     "Those answers need a T-series or E-series drawing, and asking for one "
     "early is cheaper than discovering at install that nobody has it."),
    (r"RCP|CEIL|CLNG", "ceiling",
     "No reflected ceiling plan, so plenum type, ceiling height and fixture "
     "positions are unknown -- and ceiling height is what turns a plan "
     "distance into a cable length."),
)

_NIC = re.compile(r"\bNIC\b|NOT.?IN.?CONTRACT|BY.?OTHERS", re.I)


@dataclass
class Finding:
    """One derived statement about the sheet, with what it rests on."""
    kind: str
    headline: str
    detail: str
    value: Any = None
    assumption: str = ""
    decides: str = ""


@dataclass
class SheetReading:
    plans: int
    findings: list[Finding] = field(default_factory=list)
    inventory: dict[str, int] = field(default_factory=dict)
    reach: "Reach | None" = None


def _blocks(doc: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for e in doc.modelspace():
            if e.dxftype() == "INSERT":
                out[e.dxf.name] = out.get(e.dxf.name, 0) + 1
    except Exception:  # noqa: BLE001
        pass
    return out


def _layer_feet(doc: Any, layer: str, ft_per_unit: float) -> float:
    import math  # noqa: PLC0415
    total = 0.0
    try:
        for e in doc.modelspace():
            if e.dxf.layer != layer:
                continue
            if e.dxftype() == "LINE":
                a, b = e.dxf.start, e.dxf.end
                total += math.dist((a.x, a.y), (b.x, b.y))
            elif e.dxftype() in ("LWPOLYLINE", "POLYLINE"):
                pts = [(p[0], p[1]) for p in e.get_points("xy")]
                total += sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    except Exception:  # noqa: BLE001
        return 0.0
    return total * ft_per_unit


def count_plans(tags: list[Tag], ft_per_unit: float | None) -> int:
    """How many separate drawings share this model space.

    Worth stating outright: SP-6 holds several, and a reader who assumes one
    will measure between two of them and believe the answer.
    """
    if not tags or not ft_per_unit:
        return 1
    remaining = list(tags)
    n = 0
    while remaining:
        group = one_plan(remaining, remaining[0], ft_per_unit)
        ids = {id(g) for g in group}
        remaining = [t for t in remaining if id(t) not in ids]
        n += 1
    return n


def read_the_sheet(doc: Any) -> SheetReading:
    """Everything the sheet decides, derived. Never a list of room names."""
    ft = feet_per_unit(doc) or 0.0
    tags = tags_from(doc)
    blocks = _blocks(doc)
    out: list[Finding] = []

    for pattern, kind, decides in _COUNTS:
        rx = re.compile(pattern, re.I)
        n = sum(v for k, v in blocks.items() if rx.search(k))
        if not n:
            continue
        names = sorted(k for k in blocks if rx.search(k))
        out.append(Finding(
            kind=kind + "_count", value=n,
            headline="{} {}{} drawn on the plan".format(n, kind, "" if n == 1 else "s"),
            detail=("Counted as block inserts ({}), not read off a label. The "
                    "drawing places {} of them, each at a known position."
                    .format(", ".join(names[:4]), n)),
            assumption="every {} is drawn as a block insert rather than as loose "
                       "linework".format(kind),
            decides=decides))

    layer_names = [str(l.dxf.name) for l in getattr(doc, "layers", [])]
    for (role, intent), kind, decides in _LENGTHS:
        matched = layers_with_role(layer_names, role, intent)
        feet = sum(_layer_feet(doc, name, ft) for name in matched) if ft else 0.0
        if feet < 1:
            continue
        out.append(Finding(
            kind=kind + "_ft", value=round(feet),
            headline="{:.0f} linear feet of {}".format(feet, kind.replace("_", " ")),
            detail="Summed from {} layer{} the drawing names {}, matched as "
                   "{}/{} rather than by name.".format(
                       len(matched), "" if len(matched) == 1 else "s",
                       ", ".join(matched[:3]), role, intent),
            assumption="the drawing follows the NCS layer grammar, and its declared "
                       "units are right",
            decides=decides))

    layers = {str(l.dxf.name).upper() for l in getattr(doc, "layers", [])}
    # Absence is only a finding when the file follows the standard well enough
    # for silence to mean something. `has_discipline` returns None when no
    # layer on the sheet parses as a discipline at all.
    legible = follows_standard(layer_names)
    cabling = has_discipline(layer_names, CABLING_DISCIPLINES)
    for pattern, kind, decides in _DISCIPLINES:
        # Every absence finding needs the same gate: a drawing that does not
        # follow the standard omits nothing, it merely cannot be read.
        if not legible:
            continue
        if kind == "telecom_electrical":
            if cabling is not False:
                continue
        else:
            rx = re.compile(pattern, re.I)
            if any(rx.search(name) for name in layers):
                continue
        out.append(Finding(
            kind="no_" + kind + "_layer", value=0,
            headline="No {} information on this sheet".format(kind.replace("_", "/")),
            detail="None of the {} layers is a {} layer.".format(
                len(layers), kind.replace("_", " or ")),
            assumption="the discipline would be on its own layer, as drawing "
                       "standards require",
            decides=decides))

    nic_layers = {name for name in layers if _NIC.search(name)}
    nic = 0
    try:
        nic = sum(1 for e in doc.modelspace()
                  if str(e.dxf.layer).upper() in nic_layers)
    except Exception:  # noqa: BLE001
        nic = len(nic_layers)
    if nic:
        out.append(Finding(
            kind="not_in_contract_regions", value=nic,
            headline="{} region{} hatched NOT IN CONTRACT".format(
                nic, "" if nic == 1 else "s"),
            detail="Areas the architect has marked as out of scope, drawn on the plan.",
            assumption="the NIC hatch means what it says on this sheet",
            decides="An NIC region is scope somebody else owns. Quoting it is a "
                    "giveaway; assuming it is excluded when it is not is a gap."))

    # Over EVERY label, not just room tags: the furniture plan that sits
    # beside the tenant plan carries no ROOM-TAG, so counting tags alone
    # reports one plan on a sheet that holds four.
    plans = count_plans(tags_from(doc, layer=""), ft)
    if plans > 1:
        out.append(Finding(
            kind="multiple_plans", value=plans,
            headline="{} separate drawings share this model space".format(plans),
            detail="Test-fit options, a key plan and a furniture plan can sit side "
                   "by side hundreds of feet apart in one file.",
            assumption="drawings more than {:.0f} ft apart are separate".format(
                SAME_PLAN_FT),
            decides="Any distance measured between two of them is meaningless, and "
                    "a count taken across all of them double-counts."))

    reach = cable_reach(tags, ft)
    if reach is not None:
        out.append(Finding(
            kind="cable_reach", value=round(reach.permanent_link_ft),
            headline="Furthest room is {}, a {:.0f} ft link ({})".format(
                reach.furthest, reach.permanent_link_ft, reach.verdict),
            detail=reach.note,
            assumption="cable runs orthogonally; {}% slack and a {:.0f} ft riser; "
                       "measured to room TAGS, which sit where their text sits and "
                       "not at the furthest outlet".format(
                           round((ROUTING_SLACK - 1) * 100), RISER_ALLOWANCE_FT),
            decides="Past 90 m a Cat6A drop fails certification and the fix is a "
                    "second closet, which is a five-figure surprise after signature."))

    # Always, whatever the drawing is. See `universal_findings`.
    out.extend(universal_findings(doc))

    return SheetReading(plans=plans, findings=out, reach=reach,
                        inventory=tag_census(tags))


# ======================= what any drawing yields, standard or not ===========
#
# Everything above reads an ARCHITECTURAL sheet that follows the layer
# standard. The corpus holds exactly four CAD files, and one of them --
# "BUMPER CONVEYOR MCP LOCATION.dwg" -- is neither: 75 layers named
# `$AUDIT-BAD-LAYER`, `001` and `0_...`, blocks called `eqklwmew` and `FDKSJ`,
# drawn in millimetres by a different CAD package entirely. The architectural
# derivation correctly finds nothing in it, and finding nothing in a drawing
# that states a 480V service and three breaker capacities is still a failure.
#
# These two passes do not care about layers or block names. A DIMENSION is a
# measurement the draughtsman committed to, and an engineering value written
# on a sheet is written the same way by every discipline in every country that
# uses the units.

#: Values engineers write on drawings. Each is (pattern, kind, what it decides).
_SPECS = (
    (r"(\d{2,4})\s*V(?:OLT)?\b[^.\n]{0,24}?(\d{1,3})\s*(?:KVA|KW)\b", "electrical_service",
     "The service the equipment hangs off. Voltage, phase and capacity decide "
     "whether existing switchgear can carry new load or whether a service "
     "upgrade -- the single largest line on an industrial job -- is in scope."),
    (r"(?:BREAKER|BKR|OCPD)[^.\n]{0,28}?(\d{2,4})\s*A\b", "breaker_capacity",
     "Each breaker capacity is a circuit already sized. Summed against the "
     "service capacity it says how much headroom is left before anything new "
     "can be added."),
    (r"\b(\d{1,3})\s*EA\b", "stated_quantity",
     "A quantity the drawing states outright, in the draughtsman's own units."),
    (r"\b(MCP|MDP|MCC|ATS|UPS|PDU|RTU|VFD)\b", "equipment_designation",
     "Named equipment. An MCP or MCC is a panel somebody has to reach, feed "
     "and commission, and each one is a location on the floor."),
)


def _all_text(doc):
    """Every string on the sheet, across every layout."""
    out = []
    spaces = [doc.modelspace()]
    try:
        spaces += [l for l in doc.layouts if getattr(l, "name", "") != "Model"]
    except Exception:  # noqa: BLE001
        pass
    for sp in spaces:
        for e in sp:
            if e.dxftype() not in ("TEXT", "MTEXT"):
                continue
            try:
                t = e.plain_text() if e.dxftype() == "MTEXT" else e.dxf.text
            except Exception:  # noqa: BLE001
                continue
            t = " ".join(str(t or "").split())
            if t:
                out.append(t)
    return out


def dimensions(doc) -> list[float]:
    """Every DIMENSION's measurement, in drawing units.

    A dimension is not inferred -- it is a length the draughtsman measured and
    committed to on the sheet, which makes it the most reliable number a
    drawing carries. SP-6 has none; the conveyor sheet has eleven.
    """
    out = []
    try:
        for e in doc.modelspace():
            if e.dxftype() != "DIMENSION":
                continue
            try:
                m = float(e.get_measurement())
            except Exception:  # noqa: BLE001
                continue
            if m > 0:
                out.append(m)
    except Exception:  # noqa: BLE001
        pass
    return sorted(out)


def universal_findings(doc) -> list[Finding]:
    """Derivations that hold whatever the drawing is and whoever drew it."""
    out: list[Finding] = []
    text = _all_text(doc)
    blob = "\n".join(text)

    for pattern, kind, decides in _SPECS:
        hits = re.findall(pattern, blob, re.I)
        if not hits:
            continue
        # Print the values the way an engineer writes them. "480 130" is the
        # regex's groups; "480V 130KVA" is the sheet's own language, and the
        # finding is read by people who speak it.
        unit = {"electrical_service": ("V", "KVA"), "breaker_capacity": ("A",),
                "stated_quantity": (" EA",)}.get(kind, ("",))
        flat = []
        for h in hits:
            parts = h if isinstance(h, tuple) else (h,)
            flat.append(" ".join(
                "{}{}".format(v, unit[i] if i < len(unit) else "")
                for i, v in enumerate(parts)))
        shown = sorted(set(flat))
        lines = sorted({t for t in text if re.search(pattern, t, re.I)})
        out.append(Finding(
            kind=kind, value=shown,
            headline="{}: {}".format(kind.replace("_", " ").title(), ", ".join(shown[:4])),
            detail="Read from the drawing's own text: {}".format(
                " | ".join(lines[:3])),
            assumption="the value is written on the sheet in the usual notation",
            decides=decides))

    dims = dimensions(doc)
    if dims:
        ft = feet_per_unit(doc)
        unit, conv = ("ft", ft) if ft else ("drawing units", 1.0)
        out.append(Finding(
            kind="dimensioned_runs", value=round(dims[-1] * conv, 1),
            headline="{} dimensioned lengths, longest {:.1f} {}".format(
                len(dims), dims[-1] * conv, unit),
            detail=("Lengths the draughtsman measured and annotated, rather than "
                    "anything inferred from geometry. Longest {:.1f} {}, "
                    "shortest {:.1f} {}.").format(
                        dims[-1] * conv, unit, dims[0] * conv, unit),
            assumption="the drawing's declared units are right",
            decides="A dimensioned run is the most reliable distance a sheet "
                    "carries, and distance is what turns a plan into cable, "
                    "conduit and tray footage."))
    return out
