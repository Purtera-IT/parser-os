"""Every taught lookup asks the store with the deal the line belongs to.

The store searches deal, then pack, then global. A lookup that passes no scope
searches global only, and a lesson taught for one deal -- the Deal Kit's
one-deal-at-a-time teaching -- never fires there. 2026-09-15: typed
classification and hours (PR #139); the tier head, the deflect layer and the
site pair / role heads had the same gap.
"""
from types import SimpleNamespace

from app.core import decide as decide_mod, entity_resolution as er, task_tier_classifier as ttc
from app.core.decide import Decision


class _Recorder:
    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []  # (relation, deal_id)

    def resolve(self, *, relation, text, candidates, scope, **_):
        self.calls.append((relation, getattr(scope, "deal_id", None)))
        v = self.answers.get((relation, text, getattr(scope, "deal_id", None)))
        return Decision(verdict=v, confidence=0.97, source="store", correction_id="corr") if v and v in candidates else None

    def few_shot(self, **_):
        return []


def _with(monkeypatch, store):
    monkeypatch.delenv("SOWSMITH_DISABLE_LLM", raising=False)
    prev = decide_mod.get_store(); decide_mod.set_store(store)
    return prev


def test_a_deal_scoped_tier_lesson_fires_for_its_deal_only(monkeypatch):
    line = "Conduct site survey during regular business hours"
    store = _Recorder({(ttc.TASK_TIER_RELATION, line, "deal-1"): "parent"})
    prev = _with(monkeypatch, store)
    try:
        assert ttc._taught_tier(line, "deal-1") == "parent"
        assert ttc._taught_tier(line, "deal-2") is None
        ours = SimpleNamespace(atom_type="task", raw_text=line, value={}, review_flags=[], id="t1", project_id="deal-1")
        assert ttc.infer_task_tier_for_atom(ours)[0] == "parent"
    finally:
        decide_mod.set_store(prev)
    assert ("task_tier", "deal-1") in store.calls and ("task_tier", "deal-2") in store.calls


def test_site_role_and_pair_lookups_carry_the_deal(monkeypatch):
    store = _Recorder()
    prev = _with(monkeypatch, store)
    try:
        er.semantic_site_role_drops({"site:atl_hq", "site:charlotte_lane"}, deal_id="deal-9")
        er.semantic_site_fusion_groups({"site:atl_hq", "site:atlanta_headquarters"}, None, deal_id="deal-9")
    finally:
        decide_mod.set_store(prev)
    deals = {d for _, d in store.calls}
    assert store.calls and deals == {"deal-9"}
