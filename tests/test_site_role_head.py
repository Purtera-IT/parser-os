"""A taught site_role correction reaches the vendor-address gate."""
from types import SimpleNamespace

import pytest

from app.core import decide as decide_mod
from app.core.decide import Decision
from app.core.pm_feedback import HEAD_REGISTRY, pm_correction_to_correction
from app.core.schemas import AtomType
from app.core.site_geo_fallback import suppress_vendor_sites

_SIGNATURE = "72 Madison Avenue, New York, NY 10016"


class _Store:
    def __init__(self, taught):
        self.taught = taught

    def resolve(self, *, relation, text, candidates, **_):
        v = self.taught.get(text) if relation == "physical_site" else None
        return Decision(verdict=v, confidence=0.9, source="store") if v in candidates else None

    def few_shot(self, **_):
        return []


def _site(aid, street, city, state, zip_):
    addr = f"{street}, {city}, {state} {zip_}"
    return SimpleNamespace(
        id=aid, atom_type=AtomType.physical_site, raw_text=addr,
        value={"kind": "physical_site", "address": addr, "street_address": street,
               "city": city, "state": state, "zip": zip_, "inferred": True},
        source_refs=[], review_flags=[], entity_keys=[],
    )


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    prev = decide_mod.get_store()
    yield lambda taught: decide_mod.set_store(_Store(taught))
    decide_mod.set_store(prev)


def test_site_role_head_governs_the_relation_the_gate_asks():
    spec = HEAD_REGISTRY["site_role"]
    assert spec.relation == "physical_site"
    assert set(spec.candidates) == {"job_site", "vendor_or_billing_address"}
    corr = pm_correction_to_correction({
        "head": "site_role", "dealId": "c869c1cd", "text": _SIGNATURE,
        "newValue": "vendor_or_billing_address", "scope": "global",
    })
    assert corr.relation == "physical_site" and corr.verdict == "vendor_or_billing_address"


def test_taught_signature_address_is_dropped_and_the_job_site_kept(store):
    store({_SIGNATURE: "vendor_or_billing_address"})
    job = _site("s_sf", "30 Hotaling Pl Fl 3", "San Francisco", "CA", "94111")
    sig = _site("s_ny", "72 Madison Avenue", "New York", "NY", "10016")
    kept, dropped = suppress_vendor_sites([job, sig], project_id="c869c1cd")
    assert dropped == 1
    assert [a.id for a in kept] == ["s_sf"]
