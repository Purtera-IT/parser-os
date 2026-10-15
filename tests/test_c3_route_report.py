"""The per-type route report: which line types the base may take from the
LLM. Synthetic counts only; no model, no deal text."""
from __future__ import annotations

import re
from pathlib import Path

from ml.c3.route_report import ACTABLE, VALUE_LIGHT, routed, table

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["task", "requirement", "_keep", "pricing_line"]


def _rows():
    rows = []
    rows += [(0, 0, 0, 0.95)] * 40          # task: base right and sure
    rows += [(1, 0, 1, 0.9)] * 2            # base calls a requirement a task
    rows += [(1, 1, 0, 0.6)] * 30           # requirement: base unsure
    rows += [(2, 2, 2, 0.99)] * 25          # keep
    rows += [(3, 3, 3, 0.99)] * 30          # value-heavy: always the LLM's
    return rows


def test_a_type_is_replaced_only_when_sure_precise_and_actable():
    per = table(NAMES, _rows(), sure=0.7, bar=0.85, min_n=20)
    assert per["task"]["replace"] and per["task"]["base_n"] == 42
    assert not per["requirement"]["replace"], "too few sure lines to judge"
    assert per["_keep"]["replace"]
    assert not per["pricing_line"]["replace"] and not per["pricing_line"]["acts"]


def test_a_type_the_classifier_does_better_is_not_replaced():
    rows = [(0, 0, 0, 0.95)] * 36 + [(1, 0, 1, 0.95)] * 4 + [(0, 1, 0, 0.5)] * 0
    per = table(NAMES, rows, sure=0.7, bar=0.85, min_n=20)
    assert per["task"]["base_precision"] == 0.9 and per["task"]["classifier_precision"] == 1.0
    assert not per["task"]["replace"]


def test_routed_counts_what_the_list_takes_and_gets_wrong():
    r = routed(NAMES, _rows(), {"task", "_keep"}, 0.7)
    assert r["taken"] == 67 and r["misroute"] == 2
    assert r["lines"] == 127


def test_the_value_light_set_matches_the_live_layer():
    src = (ROOT / "app/core/typed_atom_classifier.py").read_text(encoding="utf-8")
    block = src[src.index("_c3_mode != \"off\""):]
    got = re.search(r"_VALUE_LIGHT = \{([^}]*)\}", block).group(1)
    assert set(re.findall(r'"([a-z_]+)"', got)) == set(VALUE_LIGHT)


def test_every_actable_type_is_an_atom_type():
    from app.core.schemas import AtomType

    values = {t.value for t in AtomType}
    assert {t for t in ACTABLE if t != "_keep"} <= values


def test_count_reads_a_real_batch_through_a_tiny_model():
    import torch

    from ml.c3.data import IGNORE, featurize
    from ml.c3.loop import TYPE_KEY
    from ml.c3.model import C3Model
    from ml.c3.route_report import count
    from ml.c3.schema import load_schema
    from tests.test_c3_type_head import CFG, _deal

    schema = load_schema()
    torch.manual_seed(0)
    batch = featurize(_deal(), schema)
    names, rows = count(C3Model(schema, CFG), [batch], "cpu", 0.7)
    assert names[0] and len(rows) == sum(y != IGNORE for y in batch.targets[TYPE_KEY])
    assert all(0.0 <= p <= 1.0 for *_, p in rows)


def test_keep_needs_the_stricter_bar():
    rows = [(2, 2, 0, 0.99)] * 95 + [(0, 2, 0, 0.99)] * 5
    per = table(NAMES, rows, sure=0.7, bar=0.85, min_n=20)
    assert per["_keep"]["base_precision"] == 0.95 and not per["_keep"]["replace"]
    assert table(NAMES, rows, sure=0.7, bar=0.85, min_n=20, keep_bar=0.9)["_keep"]["replace"]


def test_count_reads_another_single_answer_head():
    import torch

    from ml.c3.data import IGNORE, featurize
    from ml.c3.model import C3Model
    from ml.c3.route_report import count
    from ml.c3.schema import load_schema
    from tests.test_c3_type_head import CFG, _deal

    schema = load_schema()
    torch.manual_seed(0)
    batch = featurize(_deal(), schema)
    names, rows = count(C3Model(schema, CFG), [batch], "cpu", 0.7, "col:admission")
    assert names == [a.value for a in schema.by_key()["col:admission"].answers]
    assert len(rows) == sum(y != IGNORE for y in batch.targets.get("col:admission", []))
