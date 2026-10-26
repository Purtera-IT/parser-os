"""The SowSmith tables the trainer reads, mirrored from app/core/c3_sow.py.

ml never imports app (tests/test_c3_isolation.py), so the trainer keeps its own
copy; tests/test_sow_tables_mirror.py fails the moment the two disagree, so a
change to either side is a change to both.
"""
from __future__ import annotations

#: c3_sow.TEACHABLE: each kind of SOW decision and the super head it trains.
TEACHABLE: dict[str, tuple[str, dict[str, str] | None]] = {
    "sow_section": ("read:sow_section", None),
    "sow_keep": ("read:co_action", {"keep": "keep", "drop": "reject"}),
    "sow_party": ("read:task_owner_role", None),
    "sow_supplier": ("col:supplier", None),
}

#: c3_sow._LIST_SECTION: the Studio list a line sits in -> its sow_section answer.
LIST_SECTION: dict[str, str] = {
    "overview": "overview", "scope": "scope_of_work", "arrival_protocol": "scope_of_work",
    "equipment_in_scope": "scope_of_work", "customer_responsibilities": "customer_responsibilities",
    "customer_materials": "customer_responsibilities", "assumptions": "assumptions",
    "out_of_scope": "out_of_scope", "provider_materials": "provider_responsibilities",
    "provider_tools": "provider_responsibilities", "schedule": "timeline",
}
