"""Universal parse-quality gates — seeded into the feedback store.

Concept-level exemplars for city-field shape and task/site linking. Never
deal-specific names.
"""

from __future__ import annotations

from app.core.feedback_store import SCOPE_GLOBAL, Correction

_CITY_SHAPE_RELATION = "physical_site_city_shape"
_TASK_SITE_RELATION = "task_site_anchor"


def parse_quality_corrections() -> list[Correction]:
    return [
        Correction(
            id="city_shape_street_suffix",
            relation=_CITY_SHAPE_RELATION,
            verdict="reparse_address",
            scope=SCOPE_GLOBAL,
            exemplars=[
                "city field contains Blvd or Street or Avenue or Parkway",
                "city value looks like a street name not a municipality",
                "Highland Park Blvd listed as the city name",
            ],
            instruction=(
                "A city field must be a municipality name. Street suffix tokens "
                "(Blvd, St, Ave, Rd, Pkwy) in the city column mean the address "
                "parser should re-anchor City, ST ZIP from the full line."
            ),
            created_by="seed",
            complaint_id="seed:city_shape_street_suffix",
        ),
        Correction(
            id="task_needs_site_context",
            relation=_TASK_SITE_RELATION,
            verdict="needs_site_context",
            scope=SCOPE_GLOBAL,
            exemplars=[
                "installation task with no facility or site reference",
                "cable pull checklist line under generic SOW body",
                "field work step without site_id or section heading",
            ],
            instruction=(
                "Field install tasks without a site anchor cannot prefill Deal Kit "
                "work packages. Inherit site from section_path or roster context."
            ),
            created_by="seed",
            complaint_id="seed:task_needs_site_context",
        ),
    ]


__all__ = ["parse_quality_corrections"]
