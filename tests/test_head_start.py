"""Universal PM head-start checklist: full catalog, covered-flagging, universality."""
from app.core.open_question_resolution import universal_head_start, _HEADSTART


class _A:
    def __init__(self, t):
        self.raw_text = t
        self.normalized_text = t.lower()


def test_catalog_is_a_real_head_start():
    assert len(_HEADSTART) >= 50  # a genuine head-start, not a token list


def test_empty_deal_gets_full_checklist_all_gaps():
    hs = universal_head_start([])
    assert len(hs) == len(_HEADSTART)
    assert all(not h["covered"] for h in hs)            # nothing covered
    assert all(h["category"] and h["summary"] for h in hs)


def test_covered_topics_are_flagged_not_dropped():
    deal = [_A("Payment terms are Net 30 days"), _A("Acceptance criteria: device powers on"),
            _A("On-site contact: Jane Doe 555-1212"), _A("Insurance COI bonding required")]
    hs = universal_head_start(deal)
    assert len(hs) == len(_HEADSTART)                   # full checklist always returned
    covered = [h for h in hs if h["covered"]]
    assert len(covered) >= 3                            # the addressed topics are checked off
    gaps = [h for h in hs if not h["covered"]]
    assert len(gaps) >= 40                              # still a strong gap list to chase


def test_universal_no_per_deal_label():
    # Same checklist regardless of customer identity (universality).
    a = universal_head_start([_A("ACME Corp project in Dallas")])
    b = universal_head_start([_A("Globex LLC project in Boston")])
    assert [h["field_id"] for h in a] == [h["field_id"] for h in b]
