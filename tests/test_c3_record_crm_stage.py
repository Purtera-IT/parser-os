"""stage_at from the deal's CRM stage history (c3_record.sender.crm_marks), else the milestones."""
from c3_record import annotate, build_rows, crm_marks, crm_stage_at

TL = {"created_at": "2026-07-22T20:48:59Z", "transitions": [
    {"ts": "2026-07-22T20:48:59Z", "label": "Open- Awaiting Scope", "order": 0},
    {"ts": "2026-08-03T20:16:10Z", "label": "Submitted for Quoting ", "order": 1},
    {"ts": "2026-08-05T19:53:14Z", "label": "Decision Pending", "order": 2},
    {"ts": "2026-09-01T00:00:00Z", "label": "Closed Won: 100%", "order": 3},
    {"ts": "2026-09-02T00:00:00Z", "label": "1265210287", "order": 0}]}


def test_crm_stage_at_arrival():
    m = crm_marks(TL)
    assert crm_stage_at("2026-07-01T00:00:00Z", m) == "scoping"
    assert crm_stage_at("2026-07-30T00:00:00Z", m) == "scoping"
    assert crm_stage_at("2026-08-03T20:16:10Z", m) == "quoted"
    assert crm_stage_at("2026-08-20T00:00:00Z", m) == "quoted"
    assert crm_stage_at("2026-09-10T00:00:00Z", m) == "contracting"   # an unnamed stage id moves nothing
    assert crm_stage_at("", m) == ""


def test_short_or_missing_timeline_keeps_milestones():
    assert crm_marks(None) is None
    assert crm_marks({"transitions": TL["transitions"][:1]}) is None


def test_annotate_reads_timeline():
    docs = [{"doc_id": "a", "kind": "email", "arrival": "2026-08-10T00:00:00Z", "origin": {}}]
    rows = lambda: [{"doc_id": "a", "entered_at": "2026-08-10T00:00:00Z", "text": "hello"}]
    assert annotate(rows(), docs)[0]["stage_at"] == "scoping"
    assert annotate(rows(), docs, timeline=TL)[0]["stage_at"] == "quoted"
    got = build_rows([{"key": "k", "text": "hello", "doc_id": "a"}],
                     [{"doc_id": "a", "kind": "email", "sent_at": "2026-08-10T00:00:00Z"}], timeline=TL)
    assert got[0]["stage_at"] == "quoted"
