"""What a worksheet IS should be learned, not kept in a word list.

Measured across 165 real sheets from 38 real workbooks: **74 of them (45%)**
reach `default_scope` — the branch that means nothing recognised the sheet — and
that branch routes to SCOPE, the bucket that feeds the SOW.

An SSRS customer export landed there and contributed 48,321 `scope_item` atoms
to a deal. Not because it looked like scope, but because its columns say
"Customer Code" and "Order Date" while `_looks_like_data_header` keys on "site",
"device", "qty", "part", "serial". A word list only ever knows the words
somebody already wrote down.

A head learns the KIND of sheet from one PM judgment and recognises the next one
by meaning — and abstains when it has seen nothing like it, rather than
defaulting to the most consequential bucket available.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.decide import set_store
from app.core.feedback_store import Correction, FeedbackStore
from app.core.pm_feedback import HEAD_REGISTRY
from app.parsers.sheet_classifier import (
    SheetRole,
    classify_sheet,
    learned_sheet_role,
    sheet_exemplar,
)

# The live sheet: a report preamble, then the real 43-column header on row 11.
PREAMBLE = [
    [None] * 6,
    ["CHIPOTLE MEXICAN GRILL", "SSRS-SL-CUS001-CustomerOut", None, None, None, None],
    [None] * 6,
    [None] * 6,
    ["13186519", None, None, None, None, None],
]
HEADER = ["Customer Code INT", "Customer Code", "Customer Description",
          "Contact Name", "Order Date", "Order Status"]
DATA = [["13186519", "13186519", "CHIPOTLE MEXICAN GRILL", "A Buyer", "2025-06-01", "Shipped"]]
EXPORT_ROWS = PREAMBLE + [HEADER] + DATA * 40


def _embed(texts: list[str]) -> np.ndarray:
    """Marker embedder: two sheets are near iff they share the report code."""
    out = np.zeros((len(texts), 8), dtype=np.float32)
    for i, t in enumerate(texts):
        tl = t.lower()
        out[i, 0] = 1.0 if "ssrs" in tl or "customer code" in tl else 0.0
        out[i, 1] = 1.0 if "site" in tl or "device" in tl else 0.0
        out[i, 2] = 0.2
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n > 1e-9, n, 1.0)


@pytest.fixture
def taught():
    store = FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True)
    spec = HEAD_REGISTRY["sheet"]
    store.add(
        Correction(
            id="pm_sheet_ssrs", relation=spec.relation, verdict="reference",
            candidates=list(spec.candidates), scope="global",
            exemplars=[sheet_exemplar("SSRS-SL-CUS001-CustomerOut", EXPORT_ROWS)],
            threshold=0.5,
        )
    )
    set_store(store)
    yield store
    set_store(None)


def test_the_exemplar_is_the_sheets_identity_not_its_data() -> None:
    ex = sheet_exemplar("SSRS-SL-CUS001-CustomerOut", EXPORT_ROWS)
    assert ex.startswith("SSRS-SL-CUS001-CustomerOut")
    assert "Customer Code" in ex and "Order Date" in ex
    # 15,000 customer records would make a prototype that matches any sheet
    # mentioning those customers — a coincidence, not a kind.
    assert "CHIPOTLE MEXICAN GRILL" not in ex.split("|", 1)[1]


def test_it_finds_the_header_behind_a_report_preamble() -> None:
    """The live sheet's real header is on row 11 under a title block. A header
    search that only looks at the top sees nothing at all."""
    assert "Customer Code INT" in sheet_exemplar("x", EXPORT_ROWS)


def test_a_judged_kind_of_sheet_stops_being_scope(taught) -> None:
    c = classify_sheet("SSRS-SL-CUS001-CustomerOut", EXPORT_ROWS)
    assert c.role is SheetRole.REFERENCE
    assert c.reason.startswith("learned_sheet_role")
    assert c.suppress is True, "REFERENCE routes to DROP — never mined as scope"


def test_it_generalises_to_the_same_report_on_another_deal(taught) -> None:
    """The point of a head over a word list: next month's export, different
    rows, different deal."""
    other = PREAMBLE + [HEADER] + [["999", "999", "SOME OTHER CHAIN", "B", "2025-09-01", "Open"]] * 10
    assert classify_sheet("SSRS-SL-CUS001-CustomerOut", other).role is SheetRole.REFERENCE


def test_an_unseen_sheet_still_gets_the_old_default(taught) -> None:
    """Adding knowledge must not remove any. A sheet nothing has judged behaves
    exactly as it did before."""
    rows = [["Site", "Device", "Qty"], ["HQ", "IP Camera", "12"]]
    c = classify_sheet("Site Roster", rows)
    assert c.reason == "default_scope"
    assert c.role is SheetRole.SCOPE


def test_it_degrades_open_with_no_store() -> None:
    """A classifier that failed closed on an unreachable store would silently
    stop mining every spreadsheet on the deal."""
    set_store(None)
    assert learned_sheet_role("anything", EXPORT_ROWS) is None
    assert classify_sheet("SSRS-SL-CUS001-CustomerOut", EXPORT_ROWS).reason == "default_scope"


def test_a_verdict_outside_the_roles_is_ignored(taught) -> None:
    taught.add(
        Correction(
            id="pm_sheet_junk", relation="sheet_role", verdict="not_a_role",
            scope="global", threshold=0.5,
            exemplars=[sheet_exemplar("SSRS-SL-CUS001-CustomerOut", EXPORT_ROWS)],
        )
    )
    c = classify_sheet("SSRS-SL-CUS001-CustomerOut", EXPORT_ROWS)
    assert c.role in (SheetRole.REFERENCE, SheetRole.SCOPE)
    assert "not_a_role" not in c.reason


def test_the_head_is_registered_with_a_closed_vocabulary() -> None:
    spec = HEAD_REGISTRY["sheet"]
    assert spec.relation == "sheet_role"
    assert set(spec.candidates) == {r.value for r in SheetRole}
