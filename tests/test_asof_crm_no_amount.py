"""A run cut to a date carries no CRM amount and no stage reached after the cut
(010115: the post-cutoff v3 price reached a pre-quote PM_HANDOFF as every
question's "deal value 3,830", and the timeline read closedwon)."""
import json

from app.core.orbitbrief_envelope import PARSER_MANIFEST_SIDECAR, _load_manifest_crm

CRM = {"deal_name": "010115 Decomm", "amount": 3830,
       "stage_timeline": {"current_stage": "closedwon", "transitions": [
           {"ts": "2026-07-14T15:35:25Z", "label": "Open- Awaiting Scope"},
           {"ts": "2026-07-15T12:49:02Z", "label": "Submitted for Quoting"},
           {"ts": "2026-07-16T15:35:07Z", "label": "Decision Pending"}]}}


def _write(tmp_path, ctx):
    (tmp_path / PARSER_MANIFEST_SIDECAR).write_text(json.dumps({"context": ctx}))


def test_cut_run_drops_amount_and_later_stages(tmp_path):
    _write(tmp_path, {"crm": CRM, "as_of": "2026-07-16T15:30:00Z"})
    crm = _load_manifest_crm(tmp_path)
    assert "amount" not in crm
    assert [t["label"] for t in crm["stage_timeline"]["transitions"]] == ["Open- Awaiting Scope", "Submitted for Quoting"]
    assert crm["stage_timeline"]["current_stage"] == "Submitted for Quoting"


def test_uncut_run_keeps_crm(tmp_path):
    _write(tmp_path, {"crm": CRM})
    crm = _load_manifest_crm(tmp_path)
    assert crm["amount"] == 3830 and crm["stage_timeline"]["current_stage"] == "closedwon"
