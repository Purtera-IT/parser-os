"""Loop-until-critic-quiet: converges when the critic stops finding new items.
Mock VLM (no network) returns shrinking 'missed' sets across rounds."""
from __future__ import annotations
import json
from app.core.schematic_loop_extract import extract_until_quiet


def _make_vlm(extract_rounds, critic_rounds):
    state = {"e": 0, "c": 0}
    def vlm(prompt, img, max_tokens=3000):
        # heuristic: the critic prompt mentions "MISSING"/"missed"
        is_critic = "miss" in prompt.lower() and "already extracted" in prompt.lower()
        if is_critic:
            out = critic_rounds[min(state["c"], len(critic_rounds) - 1)]; state["c"] += 1
        else:
            out = extract_rounds[min(state["e"], len(extract_rounds) - 1)]; state["e"] += 1
        return json.dumps({"rows": out})
    return vlm


def test_converges_when_critic_goes_quiet():
    extract = [
        [{"text": "camera 1"}, {"text": "camera 2"}],   # round 1 finds 2
        [{"text": "camera 3"}],                           # round 2 finds the missed one
        [],                                               # round 3 nothing new
    ]
    critic = [
        [{"text": "camera 3"}],                           # after r1, 1 missing
        [],                                               # after r2, none missing -> converge
        [],
    ]
    r = extract_until_quiet("b64", extract_prompt="extract", critic_prompt="what is MISSING",
                            vlm=_make_vlm(extract, critic), max_rounds=5)
    texts = sorted(x["text"] for x in r.rows)
    assert texts == ["camera 1", "camera 2", "camera 3"]
    assert r.converged is True
    assert r.rounds <= 3


def test_dedups_repeated_items():
    extract = [[{"text": "a"}, {"text": "a"}, {"text": "b"}], []]
    critic = [[], []]
    r = extract_until_quiet("b64", extract_prompt="e", critic_prompt="what is MISSING",
                            vlm=_make_vlm(extract, critic), max_rounds=3)
    assert sorted(x["text"] for x in r.rows) == ["a", "b"]


def test_stops_at_max_rounds_if_never_dry():
    # critic always finds something new -> bounded by max_rounds
    extract = [[{"text": f"x{i}"}] for i in range(10)]
    critic = [[{"text": f"y{i}"}] for i in range(10)]
    r = extract_until_quiet("b64", extract_prompt="e", critic_prompt="what is MISSING",
                            vlm=_make_vlm(extract, critic), max_rounds=3)
    assert r.rounds == 3
    assert r.converged is False
