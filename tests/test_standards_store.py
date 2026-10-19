"""SOW standards: packaged file + overlay added without a deploy."""
import json

import pytest

from app.core import standards_store as S


@pytest.fixture
def overlay(tmp_path, monkeypatch):
    p = tmp_path / "overlay.json"
    monkeypatch.setenv("SOWSMITH_STANDARDS_OVERLAY_FILE", str(p))
    monkeypatch.setenv("SOWSMITH_STANDARDS_TTL_S", "0")
    return p


def test_merge_appends_gates_and_retires():
    packaged = {"av_conference_room": {"exclusions": ["{when:mount} Drywall repair or painting."]}}
    ov = {"items": [
        {"family": "av_conference_room", "section": "method_steps", "text": "Test the display.", "when": "display"},
        {"family": "av_conference_room", "section": "exclusions", "text": "Drywall repair or painting.", "status": "retired"},
        {"family": "new_family", "section": "tools", "text": "Laser level."},
        {"family": "x", "section": "bogus", "text": "Ignored item."}]}
    out = S.merge(packaged, ov)
    assert out["av_conference_room"]["method_steps"] == ["{when:display} Test the display."]
    assert out["av_conference_room"]["exclusions"] == []
    assert out["new_family"]["tools"] == ["Laser level."]
    assert "x" not in out
    assert packaged["av_conference_room"]["exclusions"]  # pure


def test_proposed_waits_for_approval_then_writer_sees_it(overlay):
    from app.core.c3_sow import purtera_standards

    t = "Provide a working wall outlet within six feet of each display."
    it = S.add_item(family="av_conference_room", section="customer_responsibilities", text=t,
                    added_by="pm@purtera-it.com", source_deal="010217")
    assert it["status"] == "proposed" and json.loads(overlay.read_text())["items"][0]["source_deal"] == "010217"
    assert t not in purtera_standards("av_conference_room").get("customer_responsibilities", [])
    S.approve_item(it["id"], by="owner")
    assert t in purtera_standards("av_conference_room")["customer_responsibilities"]
    S.retire_item(it["id"], by="owner")
    assert t not in purtera_standards("av_conference_room").get("customer_responsibilities", [])


def test_bad_items_are_refused(overlay):
    with pytest.raises(ValueError):
        S.add_item(family="av", section="nowhere", text="Something sensible.", added_by="x")
    with pytest.raises(ValueError):
        S.add_item(family="av", section="tools", text="lowercase start", added_by="x")
    import re
    with pytest.raises(re.error):
        S.add_item(family="av", section="tools", text="Ladder of six feet.", when="(", added_by="x")


def test_conflicting_writes_are_refused(overlay):
    S.add_item(family="av", section="tools", text="Ladder of eight feet.", added_by="a")
    ov, etag = S.read_overlay()
    S.add_item(family="av", section="tools", text="Stud finder for walls.", added_by="b")
    with pytest.raises(S.ConflictError):
        S.write_overlay(ov, etag)


def test_no_storage_means_packaged_only(monkeypatch):
    monkeypatch.delenv("SOWSMITH_STANDARDS_OVERLAY_FILE", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("SOWSMITH_STANDARDS_TTL_S", "0")
    packaged = json.loads(S.PACKAGED.read_text())
    assert S.load() == packaged


def test_api_round_trip(overlay):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.routes_standards import router

    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    r = c.post("/standards/items", json={"family": "relocation_move", "list": "assumptions",
                                         "text": "Elevators are available for the move.", "by": "pm@x",
                                         "deal_id": "010147", "source_line": "atm_1"})
    assert r.status_code == 200, r.text
    iid = r.json()["id"]
    g = c.get("/standards").json()
    assert "Elevators are available for the move." not in g["standards"]["relocation_move"].get("assumptions", [])
    assert [(i["id"], i["status"]) for i in g["items"]] == [(iid, "proposed")]
    assert c.post(f"/standards/items/{iid}/approve", json={"by": "owner"}).json()["status"] == "approved"
    assert "Elevators are available for the move." in c.get("/standards").json()["standards"]["relocation_move"]["assumptions"]
    assert c.post("/standards/items", json={"family": "relocation_move", "list": "nope",
                                            "text": "Valid sentence here.", "by": "x"}).status_code == 422
    assert c.post(f"/standards/items/{iid}/retire", json={"by": "pm@x"}).json()["status"] == "retired"
    assert c.post("/standards/items/std_missing/approve", json={"by": "x"}).status_code == 404
