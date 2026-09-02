"""A run must be able to say what it was asked for.

An as-of run answers "what did we know at the moment we committed" -- the most
information-rich thing the pipeline produces. But a cut run and a full run
produced envelopes that were indistinguishable, so nothing downstream could tell
which set it was looking at.

Observed on deal 010215 (2026-09-02): the Deal Artifacts page displayed 11 of 69
documents beneath a selector reading "All data - no cutoff", because the chip is
local UI state that resets on reload and the envelope recorded nothing. The 18
documents the cut had excluded rendered as "Awaiting parse" -- as though the
parser had failed on them, rather than as the deliberate answer to the question
that was asked.
"""

from __future__ import annotations

import json

from app.core.orbitbrief_envelope import _load_manifest_run_cutoff
from app.core.orbitbrief_envelope import PARSER_MANIFEST_SIDECAR


def _sidecar(tmp_path, payload):
    (tmp_path / PARSER_MANIFEST_SIDECAR).write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_it_reads_the_cutoff_the_run_was_given(tmp_path):
    d = _sidecar(tmp_path, {"context": {"as_of": "2026-08-12T18:11:39.978Z"}})
    assert _load_manifest_run_cutoff(d) == "2026-08-12T18:11:39.978Z"


def test_quote_asof_is_accepted_as_the_same_field(tmp_path):
    # The manifest writes both keys; either one means "this run was cut".
    d = _sidecar(tmp_path, {"context": {"quote_asof": "2026-08-13T15:33:00.994Z"}})
    assert _load_manifest_run_cutoff(d) == "2026-08-13T15:33:00.994Z"


def test_a_full_run_reports_no_cutoff(tmp_path):
    d = _sidecar(tmp_path, {"context": {"crm": {"deal_name": "x"}}})
    assert _load_manifest_run_cutoff(d) is None


def test_a_blank_cutoff_is_no_cutoff(tmp_path):
    # "" must not read as a cutoff at the epoch, which would exclude everything.
    d = _sidecar(tmp_path, {"context": {"as_of": "   "}})
    assert _load_manifest_run_cutoff(d) is None


def test_a_missing_or_unreadable_sidecar_is_not_a_cutoff(tmp_path):
    assert _load_manifest_run_cutoff(tmp_path) is None
    (tmp_path / PARSER_MANIFEST_SIDECAR).write_text("{not json", encoding="utf-8")
    assert _load_manifest_run_cutoff(tmp_path) is None


def test_a_non_dict_context_is_not_a_cutoff(tmp_path):
    d = _sidecar(tmp_path, {"context": "nope"})
    assert _load_manifest_run_cutoff(d) is None
