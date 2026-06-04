"""Grounded-Extractor #68/#71: the extraction logging tap must capture EVERY
extractor's output as a silver training row — regardless of the dict shape that
extractor happens to emit.

The bug this guards against: a hardcoded text-field list ("text"/"name"/…) that
silently dropped any extractor whose payload lived under a different key
(risks → ``description``, acceptance_criteria → ``criterion``,
compliance_obligations → ``obligation``). Those rows never reached the log, so
the Span/Norm heads (#71) had nothing to learn from for those relations.

The tap is now shape-agnostic: a priority list, then a longest-plain-string
fallback. These tests pin both the priority behaviour and the fallback, and
prove no taught shape is dropped. Hermetic — an injected in-memory log, no LLM.
"""

from __future__ import annotations

import app.core.multi_entity_llm as mel
from app.core.multi_entity_llm import _best_extraction_text, _log_extraction_training_rows
from app.core.training_log import TrainingLog, set_training_log


class _Atom:
    def __init__(self, project_id="dealZ"):
        self.project_id = project_id


def _capture(results: dict) -> TrainingLog:
    log = TrainingLog(":memory:")
    set_training_log(log)
    try:
        _log_extraction_training_rows(results, [_Atom()])
    finally:
        set_training_log(None)
    return log


# ── _best_extraction_text unit behaviour ────────────────────────────────
def test_priority_field_wins_over_longer_value():
    # "name" is higher priority than a longer non-text-ish field.
    it = {"name": "Acme", "note": "a much longer incidental string"}
    assert _best_extraction_text(it) == "Acme"


def test_fallback_to_longest_string_for_unknown_shape():
    # No priority field present → take the longest plain string value.
    it = {"foobar": "short", "blob": "the longest meaningful payload here"}
    assert _best_extraction_text(it) == "the longest meaningful payload here"


def test_label_and_metadata_keys_never_used_as_text():
    # category/role/type/_via are the label/metadata, not the row text.
    it = {"category": "high", "role": "owner", "_via": "plir", "type": "x"}
    assert _best_extraction_text(it) == ""


def test_description_criterion_obligation_now_captured():
    # The exact shapes the old tap silently dropped.
    assert _best_extraction_text({"description": "a risk clause"}) == "a risk clause"
    assert _best_extraction_text({"criterion": "must pass UAT"}) == "must pass UAT"
    assert _best_extraction_text({"obligation": "retain records 7y"}) == "retain records 7y"


# ── end-to-end: every taught shape produces a row ────────────────────────
def test_every_extractor_shape_is_logged():
    results = {
        "customer": "Yonah County",
        "stakeholders": [{"name": "Jane Doe", "role": "PM"}],
        "requirements": [{"text": "shall provide 20 cameras"}],
        "site_clusters": [{"canonical_name": "Main St Facility", "aliases": ["MSF"]}],
        "risks": [{"description": "schedule slip risk"}],
        "acceptance_criteria": [{"criterion": "passes factory test"}],
        "penalties": [{"description": "1% per day late"}],
        "compliance_obligations": [{"obligation": "OSHA logs retained"}],
    }
    log = _capture(results)
    # 1 customer + 7 list items = 8 rows, none dropped.
    assert log.count() == 8
    # The previously-dropped shapes are present with their real text.
    risk_rows = log.rows(relation="risks")
    assert risk_rows and risk_rows[0].raw_text == "schedule slip risk"
    crit_rows = log.rows(relation="acceptance_criteria")
    assert crit_rows and crit_rows[0].raw_text == "passes factory test"


def test_empty_and_textless_items_skipped():
    results = {
        "requirements": [
            {"text": "real requirement"},
            {"category": "meta_only"},   # no text payload → skipped
            {},                          # empty → skipped
        ],
    }
    log = _capture(results)
    assert log.count(relation="requirements") == 1


def test_tap_is_noop_without_log(monkeypatch):
    # Default-off: never raises, writes nothing.
    set_training_log(None)
    monkeypatch.delenv("SOWSMITH_TRAINING_LOG_DB", raising=False)
    _log_extraction_training_rows(
        {"requirements": [{"text": "x"}]}, [_Atom()]
    )  # must not raise
