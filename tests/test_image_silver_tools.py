"""Pure-function tests for the image-silver harvest / review-queue / import
tools (tools/harvest_image_silver.py, tools/build_image_review_queue.py,
tools/import_image_silver.py).

These tools grow the ``pdf_image_kind`` silver channel; the pipeline itself
(pdf_image_gate / pdf_image_vision) is deliberately untouched, so the tests
pin the tool-side contracts: feature text identical to the runtime gate,
guess-free caption/OCR fallbacks, PM-critical suspicion flagging, dedup, and
idempotent import."""
from __future__ import annotations

import pytest

from app.core.pdf_image_gate import gate_feature_text
from app.core.training_log import TrainingLog
from tools.build_image_review_queue import (
    culprit_score,
    ocr_token_count,
    quantity_signals,
    rank_queue,
)
from tools.harvest_image_silver import (
    Block,
    build_feature_text,
    dedup_candidates,
    image_dedup_key,
    nearest_caption_above,
    suspicious_hits,
    text_layer_near_rect,
)
from tools.import_image_silver import (
    content_row_id,
    graded_to_training_row,
)


# ── feature text: MUST equal what the runtime gate sees ─────────────


def test_feature_text_is_the_gate_feature_text():
    cap, ocr = "Figure 3: rack elevation", "48 ports\npatch panel"
    assert build_feature_text(cap, ocr) == gate_feature_text(cap, ocr)
    assert build_feature_text(cap, ocr) == (
        "caption: Figure 3: rack elevation\nocr: 48 ports\npatch panel"
    )


def test_feature_text_degenerates_to_no_context():
    assert build_feature_text("", "") == "no context"
    assert build_feature_text("  ", "\n") == "no context"


def test_feature_text_truncates_ocr_at_500():
    feat = build_feature_text("", "x" * 900)
    assert feat == "ocr: " + "x" * 500


# ── caption heuristic: nearest text ABOVE with horizontal overlap ───

_IMG = (100.0, 300.0, 400.0, 500.0)  # x0, y0, x1, y1


def test_caption_picks_nearest_block_above():
    blocks = [
        Block(100, 200, 400, 220, "Far heading"),
        Block(100, 270, 400, 290, "Figure 2: patch panel labeling"),
    ]
    assert nearest_caption_above(blocks, _IMG) == "Figure 2: patch panel labeling"


def test_caption_ignores_blocks_below_and_far_above():
    blocks = [
        Block(100, 520, 400, 540, "text below the image"),
        Block(100, 10, 400, 30, "way too far above"),
    ]
    assert nearest_caption_above(blocks, _IMG) == ""


def test_caption_requires_horizontal_overlap():
    blocks = [Block(500, 270, 700, 290, "caption of the OTHER column")]
    assert nearest_caption_above(blocks, _IMG) == ""


def test_caption_whitespace_collapsed():
    blocks = [Block(100, 270, 400, 290, "  Figure   1\n rack  ")]
    assert nearest_caption_above(blocks, _IMG) == "Figure 1 rack"


# ── OCR degradation: text layer near the image rect ─────────────────


def test_text_layer_near_rect_keeps_overlapping_drops_distant():
    blocks = [
        Block(150, 350, 350, 380, "Comm Cabinet"),
        Block(150, 480, 350, 505, "18 Total Data Outlets"),  # bottom edge pad
        Block(600, 350, 800, 380, "unrelated sidebar"),
    ]
    got = text_layer_near_rect(blocks, _IMG)
    assert "Comm Cabinet" in got
    assert "18 Total Data Outlets" in got
    assert "sidebar" not in got


def test_text_layer_empty_when_nothing_near():
    assert text_layer_near_rect([Block(0, 0, 10, 10, "corner")], _IMG) == ""


# ── suspicious flag: PM-critical vocabulary in feature text ─────────


def test_suspicious_hits_fire_on_pm_terms():
    feat = build_feature_text(
        "Exhibit B", "Performance bond required; payment terms Net 30; SLA 99.99 % uptime"
    )
    kinds = {h["kind"] for h in suspicious_hits(feat)}
    assert "requirement" in kinds
    assert "quantity" in kinds
    assert len(suspicious_hits(feat)) >= 3


def test_suspicious_hits_case_insensitive():
    assert suspicious_hits("ocr: LIQUIDATED DAMAGES apply")
    assert suspicious_hits("ocr: liquidated damages apply")


def test_no_suspicious_hits_on_benign_text():
    assert suspicious_hits("caption: company logo\nocr: PurTera") == []
    assert suspicious_hits("no context") == []


# ── dedup by image content hash ─────────────────────────────────────


def test_image_dedup_key_is_content_hash():
    assert image_dedup_key(b"abc") == image_dedup_key(b"abc")
    assert image_dedup_key(b"abc") != image_dedup_key(b"abd")
    assert len(image_dedup_key(b"abc")) == 16


def test_dedup_candidates_first_occurrence_wins():
    a = {"image_sha16": "aaaa", "pdf": "first.pdf"}
    b = {"image_sha16": "aaaa", "pdf": "second.pdf"}
    c = {"image_sha16": "cccc", "pdf": "third.pdf"}
    out = dedup_candidates([a, b, c])
    assert out == [a, c]


def test_dedup_keeps_rows_without_hash():
    rows = [{"image_sha16": ""}, {"image_sha16": ""}]
    assert dedup_candidates(rows) == rows


# ── review-queue ranking ────────────────────────────────────────────


def test_quantity_signals_match_deal_shapes():
    text = "18 Total Data Outlets, 15 - Duplex Data Outlets, qty 4, $4,500"
    got = quantity_signals(text)
    assert any("18" in q for q in got)
    assert any("15" in q for q in got)
    assert any(q.lower().startswith("qty") for q in got)
    assert any(q.startswith("$") for q in got)
    assert quantity_signals("no numbers here") == []


def test_culprit_score_ordering():
    hot = culprit_score(pm_hit_count=2, quantity_count=3, tokens=80)
    warm = culprit_score(pm_hit_count=0, quantity_count=3, tokens=80)
    cold = culprit_score(pm_hit_count=0, quantity_count=0, tokens=5)
    assert hot > warm > cold


def test_culprit_score_logged_skip_bonus():
    base = culprit_score(pm_hit_count=1, quantity_count=1, tokens=30)
    logged = culprit_score(pm_hit_count=1, quantity_count=1, tokens=30, logged_skip=True)
    assert logged == pytest.approx(base + 2.5)


def test_rank_queue_sorts_desc_and_numbers_ranks():
    rows = [
        {"feature_text": "no context", "ocr_snippet": ""},
        {"feature_text": "ocr: performance bond and liquidated damages, 18 Total Data Outlets",
         "ocr_snippet": "performance bond and liquidated damages, 18 Total Data Outlets"},
    ]
    ranked = rank_queue(rows)
    assert ranked[0]["pm_hit_count"] >= 2
    assert [r["rank"] for r in ranked] == [1, 2]
    assert ranked[0]["culprit_score"] >= ranked[1]["culprit_score"]


def test_rank_queue_logged_skip_outranks_equal_harvest_row():
    feat = "ocr: patch panel 48 ports"
    harvest = {"feature_text": feat, "ocr_snippet": feat, "logged_label": ""}
    logged = {"feature_text": feat, "ocr_snippet": feat, "logged_label": "skip"}
    ranked = rank_queue([harvest, logged])
    assert ranked[0]["logged_label"] == "skip"


# ── importer: validation + idempotency ──────────────────────────────


def _graded(label="diagram", feat="caption: Figure 1\nocr: rack elevation", **kw):
    return {"label": label, "feature_text": feat, "variety": "rack", **kw}


def test_content_row_id_deterministic_and_label_sensitive():
    a = content_row_id("diagram", "caption: x")
    assert a == content_row_id("diagram", "caption: x")
    assert a != content_row_id("chart", "caption: x")
    assert a.startswith("trn_sa_")


def test_graded_row_shape_matches_silver_channel():
    row = graded_to_training_row(_graded(deal_id="d-1", rationale="rack photo text"))
    assert row.relation == "pdf_image_kind"
    assert row.teacher == "silver_audit"          # never 'llm' / 'pm'
    assert row.label_kind == "judgment"
    assert row.masked_text == row.raw_text        # mirror _log_gate_silver
    assert row.provenance["stage"] == "silver_audit_import"
    assert row.provenance["variety"] == "rack"


def test_graded_row_rejects_unknown_label():
    with pytest.raises(ValueError):
        graded_to_training_row(_graded(label="floor_plan"))  # not a gate label


def test_graded_row_accepts_runtime_kinds_beyond_fe_six():
    # instructions/label/map are pipeline kinds (_IMAGE_KIND_CANDIDATES) the
    # silver channel logs, even though the FE review set exposes only six.
    row = graded_to_training_row(_graded(label="instructions"))
    assert row.label == "instructions"


def test_graded_row_rejects_empty_feature_text():
    with pytest.raises(ValueError):
        graded_to_training_row(_graded(feat=""))


def test_graded_row_clears_local_pseudo_deal():
    row = graded_to_training_row(_graded(deal_id="local:COPPER_001"))
    assert row.deal_id == ""


def test_import_is_idempotent_by_content(tmp_path):
    db = tmp_path / "t.db"
    log = TrainingLog(str(db))
    rows = [graded_to_training_row(_graded()),
            graded_to_training_row(_graded(label="chart", feat="ocr: revenue by quarter"))]
    log.add_many(rows)
    # Re-import the SAME content: ids collide -> INSERT OR REPLACE -> no growth.
    log.add_many([graded_to_training_row(_graded()),
                  graded_to_training_row(_graded(label="chart", feat="ocr: revenue by quarter"))])
    assert log.count(relation="pdf_image_kind") == 2
    assert log.count(relation="pdf_image_kind", teacher="silver_audit") == 2
