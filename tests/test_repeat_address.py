"""010358 (2026-10-29): a bare repeat of an address takes the heads' answer on its full-context copy, upstream."""
import pytest

from app.core import rule_ledger
from app.core.repeat_address import RULE_ID, follow_full_copies


def _r(key, text, doc="d1", order=0.0, **kw):
    return {"key": key, "atom_id": key, "text": text, "doc_id": doc, "order": order, **kw}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("SOWSMITH_RULE_MODES", raising=False)
    rule_ledger.reset()
    yield
    rule_ledger.reset()


def _rows():
    return [
        _r("s1", "Scott Morgan | 1361 N Glenville Dr | scott@traversa.net | 214-707-0631", "d0", 0),
        _r("b1", "e | scott@traversa.net", "d1", 7),
        _r("p1", "1361 N Glenville Dr", "d1", 9),
        _r("c1", "Richardson TX 75081", "d1", 10),
        _r("p2", "1361 N Glenville Dr", "d2", 29),
        _r("c2", "Richardson TX 75081", "d2", 30),
        _r("x1", "4290 Thurman Road, Conley, GA 30288", "d3", 1),
        _r("x2", "Richardson TX 75081", "d3", 5),
    ]


def test_bare_street_and_its_city_line_take_the_contact_lines_answer():
    types = {"s1": ["stakeholder", 0.909], "b1": ["stakeholder", 0.8], "p1": ["stakeholder", 0.88],
             "c1": ["stakeholder", 0.76], "p2": ["physical_site", 0.997], "c2": ["physical_site", 0.9],
             "x1": ["physical_site", 0.99], "x2": ["physical_site", 0.9]}
    deal = {}
    assert sorted(follow_full_copies(types, _rows(), deal)) == ["c2", "p2"]
    assert types["p2"] == ["stakeholder", 0.909] and types["c2"] == ["stakeholder", 0.909]
    assert deal["own_reads"]["p2"] == ["physical_site", 0.997] and deal["follows_copy"]["p2"]["copy"] == "s1"
    # a site with no full-context copy, and a city line with no followed street before it, keep their reads
    assert types["x1"][0] == "physical_site" and types["x2"][0] == "physical_site"
    rec = rule_ledger.fires()[RULE_ID]
    assert rec["stage"] == "upstream" and rec["fires"] == 4 and rec["agree_with_head"] == 2 and rec["disagree"] == 2


def test_shadow_logs_and_changes_nothing(monkeypatch):
    monkeypatch.setenv("SOWSMITH_RULE_MODES", f"{RULE_ID}:shadow")
    types = {"s1": ["stakeholder", 0.9], "p2": ["physical_site", 0.997], "c2": ["physical_site", 0.9]}
    assert follow_full_copies(types, _rows()) == []
    assert types["p2"][0] == "physical_site"
    assert rule_ledger.fires()[RULE_ID]["disagree"] == 2


def test_copies_that_disagree_decide_nothing():
    rows = [_r("s1", "Jane Doe | 100 Main St Suite 4 | jane@acme.com"),
            _r("t1", "100 Main St Suite 4 | Tier: Large", "sheet", 1, sheet="Sites", table_id="sheet:Sites"),
            _r("p1", "100 Main St Suite 4", "d2", 3)]
    types = {"s1": ["stakeholder", 0.9], "t1": ["physical_site", 0.95], "p1": ["physical_site", 0.99]}
    assert follow_full_copies(types, rows) == [] and types["p1"][0] == "physical_site"


def test_a_roster_row_is_never_the_bare_copy_and_can_be_the_full_one():
    rows = [_r("t1", "Store 12 | 100 Main St Ste 4 | Austin TX", "sheet", 1, sheet="Sites", table_id="sheet:Sites"),
            _r("r2", "100 Main St Ste 4", "sheet", 2, sheet="Sites", table_id="sheet:Sites"),
            _r("p1", "100 Main St Ste 4", "d2", 3)]
    types = {"t1": ["physical_site", 0.95], "r2": ["stakeholder", 0.6], "p1": ["deal_metadata", 0.7]}
    assert follow_full_copies(types, rows) == ["p1"]
    assert types["r2"][0] == "stakeholder" and types["p1"][0] == "physical_site"


def test_only_address_answers_move_and_signatory_gives_stakeholder():
    rows = [_r("g1", "Pat Lee, VP | 9 Elm Street | pat@x.com"), _r("p1", "9 Elm Street", "d2", 1),
            _r("q1", "9 Elm Street", "d3", 1)]
    types = {"g1": ["signatory", 0.8], "p1": ["physical_site", 0.99], "q1": ["small_talk", 0.9]}
    assert follow_full_copies(types, rows) == ["p1"]
    assert types["p1"][0] == "stakeholder" and types["q1"][0] == "small_talk"
