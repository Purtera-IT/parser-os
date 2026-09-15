"""An as-of run's envelope carries only the deal history before its cutoff (live 010162)."""
from app.core import orbitbrief_envelope as env

EVENTS = [
    {"date": "2026-07-30T18:30:00", "type": "PRICING_REQUESTED", "summary": "fixed fee per site"},
    {"date": "2026-08-06T10:41:02", "type": "SCHEDULED", "summary": "Technicians are onsite"},
    {"date": "2026-08-20T21:15:50.755000", "type": "SOW_SIGNED", "summary": "Signed SOW received"},
    {"type": "NOTE", "summary": "undated"},
]


def _patch(monkeypatch, quote_asof):
    monkeypatch.setattr(env._timeline, "events", lambda pid: [dict(e) for e in EVENTS])
    monkeypatch.setattr(env._timeline, "quote_asof", lambda pid: quote_asof)


def test_events_after_the_run_cutoff_are_dropped(monkeypatch):
    _patch(monkeypatch, "2026-08-03T16:31:00")
    out = env._deal_timeline_section("deal", [], run_cutoff="2026-08-03T16:10:45.000Z")
    assert [e["type"] for e in out["events"]] == ["PRICING_REQUESTED"]
    assert out["quote_asof"] is None, "the quote came after the cutoff, so this run does not know it"
    assert out["known"] is True


def test_a_quote_before_the_cutoff_is_kept(monkeypatch):
    _patch(monkeypatch, "2026-07-31T09:00:00")
    out = env._deal_timeline_section("deal", [], run_cutoff="2026-08-03T16:10:45Z")
    assert out["quote_asof"] == "2026-07-31T09:00:00"


def test_a_full_run_keeps_the_whole_history(monkeypatch):
    _patch(monkeypatch, "2026-08-03T16:31:00")
    out = env._deal_timeline_section("deal", [], run_cutoff=None)
    assert len(out["events"]) == 4 and out["quote_asof"] == "2026-08-03T16:31:00"
