"""The contrastive trainer was never type-specific. Only its loader was.

`runpod_detector/train_contrastive_encoder_gpu.py` is the architecture that
breaks the ceiling a frozen space imposes (its own measurements: kNN on frozen
~0.65, a classifier head ~0.82, a re-sorted space beats both). It read
`WHERE relation='atom_type'` and nothing else, so every other head a labeler
feeds was stuck on the frozen space no matter how much gold arrived.
"""
from __future__ import annotations

import importlib
import os
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


_SEQ = iter(range(1, 10_000))


def _db(tmp_path: Path) -> Path:
    p = tmp_path / f"_training_test_{next(_SEQ)}.db"
    con = sqlite3.connect(p)
    con.execute(
        "CREATE TABLE IF NOT EXISTS training_rows (relation TEXT, label TEXT, raw_text TEXT, "
        "masked_text TEXT, deal_id TEXT, teacher TEXT)"
    )
    con.executemany(
        "INSERT INTO training_rows VALUES (?,?,?,?,?,?)",
        [
            ("reads:blocked_on", "us", "a reseller asked whether we can do it", None, "d1", "human"),
            ("reads:blocked_on", "partner", "we asked whether they can do it", None, "d2", "human"),
            ("about", "account", "it could lead to many more of the same", None, "d1", "human"),
            ("atom_type", "deal_metadata", "here are the details", None, "d1", "human"),
        ],
    )
    con.commit()
    con.close()
    return p


def _load(relation: str, db: Path):
    sys.path.insert(0, str(ROOT / "runpod_detector"))
    os.environ["RELATION"] = relation
    os.environ["SOWSMITH_TRAINING_LOG_DB"] = str(db)
    mod = importlib.import_module("train_contrastive_encoder_gpu")
    importlib.reload(mod)
    try:
        return mod.load(), mod
    finally:
        sys.path.remove(str(ROOT / "runpod_detector"))


@pytest.mark.parametrize("relation, expected", [
    ("reads:blocked_on", {"us", "partner"}),
    ("about", {"account"}),
])
def test_any_relation_can_be_trained_and_its_label_is_its_class(relation, expected, tmp_path):
    pytest.importorskip("sentence_transformers")
    rows, mod = _load(relation, _db(tmp_path))
    assert {r[2] for r in rows} == expected, "the label IS the class outside the type taxonomy"
    assert mod.RELATION == relation
    assert not mod.IS_TYPE


def test_two_heads_do_not_overwrite_each_others_artifacts(tmp_path):
    pytest.importorskip("sentence_transformers")
    _, a = _load("about", _db(tmp_path))
    dir_a = a.RUN_DIR
    _, b = _load("reads:blocked_on", _db(tmp_path))
    assert dir_a != b.RUN_DIR
    assert ":" not in b.RUN_DIR, "a relation with a colon must still be a usable path"


def test_the_type_head_still_goes_through_its_taxonomy(tmp_path):
    pytest.importorskip("sentence_transformers")
    rows, mod = _load("atom_type", _db(tmp_path))
    assert mod.IS_TYPE
    assert mod.RUN_DIR.endswith(mod.LABEL_MODE)
    # deal_metadata goes through the facet taxonomy, so the class it trains
    # under is a facet -- never the raw label.
    assert rows, "the type loader still reads its rows"
    assert all(r[2] != "deal_metadata" for r in rows)


def test_an_inherited_os_variable_does_not_crash_the_trainer(tmp_path, monkeypatch):
    # On Windows `TEMP` is the temp DIRECTORY. The trainer read
    # float(os.environ["TEMP"]) and died before loading a single row:
    #   ValueError: could not convert string to float: 'D:\temp'
    # An inherited variable is not a configuration choice.
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("TEMP", r"D:\temp")
    monkeypatch.setenv("BATCH", "not a number")
    _, mod = _load("about", _db(tmp_path))
    assert mod.TEMP == 0.07
    assert mod.BATCH == 128
    monkeypatch.setenv("SUPCON_TEMP", "0.05")
    _, mod2 = _load("about", _db(tmp_path))
    assert mod2.TEMP == 0.05
