"""What a lesson is keyed on: the shape of the work, not the words used for it.

A taught head that keys on wording reproduces the deal it was taught on and
misses the next customer's phrasing (measured cross-deal transfer: ~6%). This
module is the alternative key. It is built from the structured work order
(:mod:`app.core.work_order`) -- action, object, unit, count, sites and the
deal-level facts that change how work is priced -- and deliberately drops the
customer, the document title and the sentence itself.

Two parts, per correctable field (see ``docs/LESSON_KEYS.md``):

* **gate** -- structural facts that must AGREE (where both sides state them)
  before a lesson may fire at all. A lesson about installing cameras on site
  never fires on activating camera licences remotely, however alike the two
  sentences read. This is what keeps deal C (same wording, different work)
  unchanged.
* **key text** -- a canonical ``facet:value`` string embedded by the store in
  place of the sentence. It carries the gate facets plus soft facets (action,
  size band, site band) so the nearest lesson still wins among the ones the
  gate admits. This is what lets deal B (same work, different wording) match.

No vocabulary lists. Values are normalised structurally (lower-case, head noun,
light stemming) from whatever the work order extracted; nothing here knows the
name of any piece of equipment.

Behind a flag: ``SOWSMITH_LESSON_KEY=work_shape`` switches the key. The default
``wording`` keeps today's behaviour byte-identical.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

KEY_MODE_ENV = "SOWSMITH_LESSON_KEY"
MODE_WORDING = "wording"
MODE_WORK_SHAPE = "work_shape"

#: Correction ``relations`` entries this module owns.
REL_SHAPE = "work_shape"
REL_GATE = "work_shape_gate"
REL_FIELD = "work_shape_field"

_WORD_RE = re.compile(r"[a-z0-9]+")


def lesson_key_mode() -> str:
    raw = os.environ.get(KEY_MODE_ENV, "").strip().lower()
    return MODE_WORK_SHAPE if raw == MODE_WORK_SHAPE else MODE_WORDING


def _stem(word: str) -> str:
    w = word.lower()
    for suffix in ("ies", "es", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return w


def head_noun(phrase: Any) -> str:
    """The last word of a noun phrase, stemmed: "security cameras" -> "camera".

    Structural, not lexical: a modifier ("security", "ceiling-mounted") is
    wording, the head noun is the thing."""
    words = _WORD_RE.findall(str(phrase or "").lower())
    words = [w for w in words if not w.isdigit()]
    return _stem(words[-1]) if words else ""


def _first_word(phrase: Any) -> str:
    words = [w for w in _WORD_RE.findall(str(phrase or "").lower()) if not w.isdigit()]
    return _stem(words[0]) if words else ""


def size_band(count: Any) -> str:
    """Order of magnitude, never the number: 3 and 7 are the same job size."""
    try:
        n = float(count)
    except (TypeError, ValueError):
        return ""
    if n <= 0 or math.isnan(n):
        return ""
    return f"1e{int(math.floor(math.log10(n)))}"


def site_band(site_count: Any) -> str:
    try:
        n = float(site_count)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    return "single" if n <= 1 else "multi"


def _flag(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, str):
        return "yes" if v.strip().lower() in ("1", "true", "yes", "y") else "no"
    return "yes" if bool(v) else "no"


def _norm(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(v or "").strip().lower()).strip("_")


@dataclass(frozen=True)
class WorkShape:
    """One unit of work, described by what is done -- never by who or how it was said.

    Empty string means "not stated"; an unstated facet never blocks a gate."""

    action: str = ""
    object: str = ""
    unit: str = ""
    size_band: str = ""
    site_band: str = ""
    after_hours: str = ""
    no_onsite_hands: str = ""
    customer_supplies_equipment: str = ""
    delivery_model: str = ""
    billing_type: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "WorkShape":
        d = d or {}
        return cls(**{k: str(d.get(k) or "") for k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


def from_work_line(
    line: dict,
    work_order: dict | None = None,
    *,
    billing_type: str = "",
    delivery_model: str = "",
) -> WorkShape:
    """Shape of one work-order line plus its deal-level facts.

    ``line`` is a ``work_lines[]`` entry (work/object/count/unit, optional
    action); ``work_order`` carries site_count, after_hours, no_onsite_hands and
    customer_supplies_equipment exactly as :data:`app.core.work_order.WORK_ORDER_PROMPT`
    returns them."""
    wo = work_order or {}
    action = line.get("action") or _first_word(line.get("work"))
    unit = line.get("unit") or line.get("object")
    return WorkShape(
        action=_stem(str(action)) if action else "",
        object=head_noun(line.get("object")),
        unit=head_noun(unit),
        size_band=size_band(line.get("count")),
        site_band=site_band(wo.get("site_count")),
        after_hours=_flag(wo.get("after_hours")),
        no_onsite_hands=_flag(wo.get("no_onsite_hands")),
        customer_supplies_equipment=_flag(wo.get("customer_supplies_equipment")),
        delivery_model=_norm(delivery_model or line.get("delivery_model") or wo.get("delivery_model")),
        billing_type=_norm(billing_type or wo.get("billing_type")),
    )


def from_atom_value(value: dict | None) -> WorkShape | None:
    """Shape of a task atom minted by the work-order stage, or None when the
    atom carries no structure (a verbatim span has wording, not shape)."""
    v = value or {}
    if not (v.get("object") or v.get("unit")):
        return None
    qc = v.get("quote_context") if isinstance(v.get("quote_context"), dict) else {}
    return from_work_line(
        {"work": v.get("text"), "object": v.get("object"), "unit": v.get("unit"), "count": v.get("count")},
        {
            "site_count": v.get("site_count"),
            "after_hours": v.get("after_hours"),
            "no_onsite_hands": v.get("no_onsite_hands"),
            "customer_supplies_equipment": v.get("customer_supplies_equipment"),
        },
        delivery_model=str((qc or {}).get("delivery_model") or ""),
        billing_type=str(v.get("billing_type") or ""),
    )


@dataclass(frozen=True)
class FieldKey:
    """What one correctable field's lesson matches on, and what it ignores."""

    gate: tuple[str, ...]
    soft: tuple[str, ...]


# Field-by-field key spec. The rationale for every entry is in
# docs/LESSON_KEYS.md; change the two together. Facets not named for a field
# are ignored by that field's lesson.
_COMMON_GATE = ("delivery_model", "no_onsite_hands", "customer_supplies_equipment")
FIELD_KEYS: dict[str, FieldKey] = {
    # How long one visit takes depends on WHAT is worked on and HOW it is
    # delivered, not on how many sites there are.
    "hours_per_visit": FieldKey(
        gate=("object", "unit", "after_hours", *_COMMON_GATE),
        soft=("action", "size_band"),
    ),
    # How many visits depends on the site pattern and the working window, not
    # on the equipment's name.
    "visits": FieldKey(
        # Object is gated too: without it a visits lesson on laptops changed
        # 17 unrelated fixture deals (cameras, cabling, TVs) whose site pattern
        # happened to agree -- a key loose enough to change everything.
        gate=("object", "site_band", "after_hours", *_COMMON_GATE),
        # Size is ignored: the lesson is a rate per site, so job size is
        # already divided out.
        soft=("action",),
    ),
    # What a count counts depends on the object and the unit it is priced in.
    "units": FieldKey(
        gate=("object", "unit", "delivery_model"),
        # Size is ignored: units per stated count does not scale with the count.
        soft=("action",),
    ),
}
DEFAULT_FIELD_KEY = FieldKey(
    gate=("object", "unit", *_COMMON_GATE),
    soft=("action", "size_band", "site_band", "after_hours"),
)


def field_key(field_name: str) -> FieldKey:
    return FIELD_KEYS.get(field_name, DEFAULT_FIELD_KEY)


def key_text(shape: WorkShape, field_name: str) -> str:
    """The canonical string the store embeds for this field's lesson.

    Stated facets only, in a fixed order, as single ``facet:value`` tokens --
    no sentence, no customer, no count."""
    fk = field_key(field_name)
    d = shape.as_dict()
    parts = [f"field:{field_name}"]
    for facet in (*fk.gate, *fk.soft):
        val = d.get(facet) or ""
        if val:
            parts.append(f"{facet}:{val}")
    return " ".join(parts)


def gate_holds(taught: dict | None, query: dict | None, gate: list[str] | tuple[str, ...]) -> bool:
    """Do the gate facets agree wherever BOTH shapes state them?

    A query with no shape at all fails the gate: a shape-keyed lesson has
    nothing to say about a line whose work was never described."""
    if not gate:
        return True
    if not taught or not query:
        return False
    for facet in gate:
        a = str(taught.get(facet) or "")
        b = str(query.get(facet) or "")
        if a and b and a != b:
            return False
    return True


def correction_gate_holds(correction_relations: dict | None, query_relations: dict | None) -> bool:
    """Store-side check: a correction carrying a work-shape gate fires only on a
    query whose work shape passes it. Corrections without a gate are untouched."""
    rel = correction_relations or {}
    gate = rel.get(REL_GATE)
    if not gate:
        return True
    q = (query_relations or {}).get(REL_SHAPE)
    return gate_holds(rel.get(REL_SHAPE), q, list(gate))


def lesson_relations(shape: WorkShape, field_name: str) -> dict[str, Any]:
    return {
        REL_SHAPE: shape.as_dict(),
        REL_GATE: list(field_key(field_name).gate),
        REL_FIELD: field_name,
    }


__all__ = [
    "KEY_MODE_ENV",
    "MODE_WORDING",
    "MODE_WORK_SHAPE",
    "FIELD_KEYS",
    "FieldKey",
    "WorkShape",
    "correction_gate_holds",
    "field_key",
    "from_atom_value",
    "from_work_line",
    "gate_holds",
    "head_noun",
    "key_text",
    "lesson_key_mode",
    "lesson_relations",
    "site_band",
    "size_band",
]
