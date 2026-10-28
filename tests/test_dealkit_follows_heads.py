"""Deal Kit follows the heads (AUDIT_dealkit_heads items 7, 12): a rule's own output is never logged
as a label nor retrained into a "neural head", and facility's "<City> Office" naming stays off until
the facility head is promoted."""
import types

import pytest

from app.core import hardware_evidence_backfill as hw
from app.core import quote_context_head as qc
from app.core import site_facility_head as sf
from app.core.training_log import RULE_TAUGHT_RELATIONS, TEACHER_RULE, TEACHER_STORE
from app.learning import retrain

S = types.SimpleNamespace


@pytest.fixture
def logged(monkeypatch):
    rows = []
    for mod in (qc, sf, hw):
        monkeypatch.setattr(mod, "log_rows", lambda rs, _r=rows: _r.extend(rs) or len(rs))
    monkeypatch.setattr("app.learning.head_registry.get_head_registry", lambda: None)
    return rows


def _task():
    return S(id="t1", atom_type="task", raw_text="Install and mount 4 APs", value={"is_quote_line": True})


def _site():
    return S(id="s1", atom_type="physical_site", raw_text="Office, Austin TX",
             value={"city": "Austin", "name": "Austin office"}, review_flags=[], entity_keys=["site:a"])


def _city_site():
    # no name: the rule falls back to the city, a trainable rule decision
    return S(id="s2", atom_type="physical_site", raw_text="Austin TX", value={"city": "Austin"},
             review_flags=[], entity_keys=["site:b"])


def _hw(monkeypatch):
    monkeypatch.setattr(hw, "_mint_bom_line", lambda **k: S(id="bom", atom_type="bom_line", raw_text=k["sku"]))
    return [S(id="p1", atom_type="scope_item", raw_text="Install 4 x E7 APs in the lobby", value={})]


def test_rule_output_is_not_logged_by_default(monkeypatch, logged):
    monkeypatch.delenv("SOWSMITH_LOG_RULE_ROWS", raising=False)
    atoms, n = qc.annotate_quote_context([_task()], project_id="d1")
    assert n == 1 and "quote_context_training_row" not in atoms[0].review_flags
    sf.annotate_site_facility_labels([_city_site()], project_id="d1")
    _, minted = hw.backfill_hardware_bom_lines(_hw(monkeypatch), project_id="d1")
    assert minted >= 1
    assert logged == []


def test_rule_output_logged_on_request_is_teacher_rule_never_store(monkeypatch, logged):
    monkeypatch.setenv("SOWSMITH_LOG_RULE_ROWS", "on")
    qc.annotate_quote_context([_task()], project_id="d1")
    sf.annotate_site_facility_labels([_city_site()], project_id="d1")
    hw.backfill_hardware_bom_lines(_hw(monkeypatch), project_id="d1")
    rels = {r.relation for r in logged}
    assert rels == set(RULE_TAUGHT_RELATIONS)
    assert all(r.teacher == TEACHER_RULE and r.teacher != TEACHER_STORE for r in logged)


def test_retrain_all_skips_rule_taught_relations_unless_asked(monkeypatch):
    monkeypatch.delenv("SOWSMITH_RETRAIN_RULE_RELATIONS", raising=False)
    skip = retrain.non_training_relations()
    assert RULE_TAUGHT_RELATIONS <= skip and retrain.NON_TRAINING_RELATIONS <= skip
    monkeypatch.setenv("SOWSMITH_RETRAIN_RULE_RELATIONS", "on")
    assert not (RULE_TAUGHT_RELATIONS & retrain.non_training_relations())


def test_retrain_all_never_trains_a_rule_relation(monkeypatch):
    monkeypatch.delenv("SOWSMITH_RETRAIN_RULE_RELATIONS", raising=False)
    trained = []
    monkeypatch.setattr(retrain, "retrain_relation", lambda log, rel, *a, **k: trained.append(rel) or S(relation=rel))
    log = S(rows=lambda: [S(relation="quote_delivery_model"), S(relation="site_facility_label"),
                          S(relation="hardware_evidence_line"), S(relation="task_hours")])
    try:
        retrain.retrain_all(log, S(), lambda texts: __import__("numpy").ones((len(texts), 4)), record_history=False)
    except Exception:
        pass  # the history / report tail is not under test; the relation list is
    assert "task_hours" in trained
    assert not (set(trained) & RULE_TAUGHT_RELATIONS)


class _CityHead:
    def classify(self, vec, candidates):
        return S(verdict=sf.CITY_OFFICE, route_llm=False, confidence=0.9)


def _with_city_champion(monkeypatch):
    reg = S(champion=lambda rel: (_CityHead(), {}))
    monkeypatch.setattr("app.learning.head_registry.get_head_registry", lambda: reg)
    monkeypatch.setattr("app.core.embedding_retrieval.embed_texts", lambda texts: [[1.0]])


def test_city_office_name_is_not_invented_until_facility_is_promoted(monkeypatch):
    _with_city_champion(monkeypatch)
    monkeypatch.delenv("SOWSMITH_C3_FACILITY_PROMOTED", raising=False)
    d = sf.decide_site_facility_label(_site())
    assert d.facility_name == "Austin office" and "Austin Office" != d.facility_name
    monkeypatch.setenv("SOWSMITH_C3_FACILITY_PROMOTED", "on")
    assert sf.decide_site_facility_label(_site()).facility_name == "Austin Office"
