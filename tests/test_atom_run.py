"""The atom run's three fixes, pinned.

A1  one typing vocabulary -- the three diverged regex copies gave 4.0% of
    real held-out sentences a different TYPE depending on their format.
A2  every harvested training row carries decide_text_version, because a head
    trained on bare rows and served decorated text is silently out of
    distribution -- and unversioned rows made that undetectable.
A3  the cascade's contrastive slot can serve a fine-type head in "typed"
    mode: value-light assignments only, strict gate, guess-free fallthrough.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── A1: the families are one object ─────────────────────────────────────


def test_the_three_typing_families_are_one_object() -> None:
    """Aliases, not copies. Copies drift; an alias cannot."""
    from app.core import atom_typing
    from app.parsers import markdown_parser, pptx_parser, universal_parsers

    for module in (markdown_parser, pptx_parser, universal_parsers):
        assert module._EXCLUSION_RE is atom_typing.EXCLUSION_RE, module.__name__
        assert module._ASSUMPTION_RE is atom_typing.ASSUMPTION_RE, module.__name__
        assert module._CONSTRAINT_RE is atom_typing.CONSTRAINT_RE, module.__name__


@pytest.mark.parametrize(
    "sentence, expected",
    [
        # each previously typed by SOME families and missed by others
        ("Customer will cancel the badge reader order.", "exclusion"),      # markdown-only vocab
        ("Work is on hold pending permit approval.", "exclusion"),          # markdown-only vocab
        ("Customer provides all mounting hardware.", "assumption"),         # universal-only vocab
        ("Final acceptance occurs at closeout.", "constraint"),             # markdown-only vocab
        ("Regulatory compliance review is mandatory.", "constraint"),       # universal-only vocab
    ],
)
def test_the_union_types_the_same_everywhere(sentence: str, expected: str) -> None:
    """The A1 measurement: 32/800 holdout sentences typed differently by
    format. The union means a sentence can only GAIN a specific type --
    never fall to scope_item where a sibling format would have typed it."""
    from app.core.atom_typing import classify_prose
    from app.parsers.pptx_parser import _classify_text
    from app.parsers.universal_parsers import _classify

    assert str(classify_prose(sentence)).split(".")[-1] == expected
    assert str(_classify(sentence)).split(".")[-1] == expected
    assert str(_classify_text(sentence, is_heading=False)).split(".")[-1] == expected


# ── A2: the representation version travels ──────────────────────────────


def test_the_harvest_tap_stamps_the_decide_text_version() -> None:
    import inspect

    from app.core import typed_atom_classifier as tac

    assert tac.DECIDE_TEXT_VERSION >= 2
    source = inspect.getsource(tac)
    assert '"decide_text_version": DECIDE_TEXT_VERSION' in source, (
        "the harvest tap must stamp the representation version, or v0 bare "
        "rows and v2 decorated rows are indistinguishable forever"
    )


def test_the_multitask_table_carries_and_reports_the_version(tmp_path) -> None:
    import json
    import sqlite3

    from app.learning.multitask_table import assemble

    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE training_rows (relation TEXT, label TEXT, raw_text TEXT,"
        " deal_id TEXT, split TEXT, teacher TEXT, provenance TEXT)"
    )
    conn.executemany(
        "INSERT INTO training_rows VALUES (?,?,?,?,?,?,?)",
        [
            ("atom_type", "exclusion", "Mid-turn jumpers are excluded.",
             "d1", "train", "llm",
             json.dumps({"decide_text_version": 2})),
            ("atom_type", "constraint", "Escort access required at dock.",
             "d2", "train", "llm", None),  # legacy, unversioned
        ],
    )
    conn.commit()
    conn.close()

    table = assemble([db])
    versions = {r.repr_version for r in table.rows}
    assert versions == {0, 2}
    assert "v0" in table.summary() and "v2" in table.summary()


# ── A3: the typed mode in the contrastive slot ──────────────────────────


class _Atom:
    def __init__(self, text: str) -> None:
        self.raw_text = text
        # the cascade only considers atoms whose current type is in
        # _PROMOTABLE_FROM ({scope_item, entity, customer_instruction})
        self.atom_type = "scope_item"
        self.value = {}
        self.id = f"atm_{abs(hash(text)) % 99999}"
        self.project_id = "p"


class _FakeTypedHead:
    """Stands in for the built store: answers what the test dictates."""

    mode = "typed"

    def __init__(self, answers) -> None:
        self._answers = answers

    def classify_batch(self, texts):
        return [self._answers.get(idx) for idx in range(len(texts))]


def _run_cascade(monkeypatch, atoms, answers):
    from app.core import typed_atom_classifier as tac

    # The suite runs with SOWSMITH_DISABLE_LLM=1, which short-circuits the
    # WHOLE cascade before any deflect layer -- unset it here and stub the LLM
    # batch call instead, so survivors fall through with their types intact,
    # which is exactly the guess-free contract under test.
    monkeypatch.delenv("SOWSMITH_DISABLE_LLM", raising=False)
    monkeypatch.setenv("SOWSMITH_CONTRASTIVE_TYPE", "1")
    monkeypatch.setattr(tac, "_classify_batch", lambda batch: {})
    monkeypatch.setattr(
        "app.core.contrastive_type_knn.load_promoted",
        lambda *a, **k: _FakeTypedHead(answers),
    )
    return tac.classify_atoms(atoms)


def test_typed_mode_assigns_value_light_and_only_value_light(monkeypatch) -> None:
    """The head's best classes (bom_line 94%) are value-heavy ON PURPOSE:
    the LLM synthesises their value payloads, so the label alone is not the
    deliverable. A value-heavy answer must fall through untouched."""
    atoms = [
        _Atom("Mid-turn jumpers are excluded from this order."),
        _Atom("Technician #1 - TV Install | $98.00 | Per Hour"),
        _Atom("Escort access is required at the dock."),
    ]
    _run_cascade(monkeypatch, atoms, {
        0: ("exclusion", 0.99),     # value-light -> assigned
        1: ("bom_line", 0.99),      # value-heavy -> MUST fall through
        2: None,                     # abstain -> falls through
    })
    assert str(atoms[0].atom_type).split(".")[-1] == "exclusion"
    assert atoms[1].atom_type == "scope_item", "value-heavy must reach the LLM"
    assert atoms[2].atom_type == "scope_item", "abstain must reach the LLM"


def test_flag_off_is_byte_identical(monkeypatch) -> None:
    from app.core import typed_atom_classifier as tac

    monkeypatch.delenv("SOWSMITH_CONTRASTIVE_TYPE", raising=False)
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("must not load with the flag off")

    monkeypatch.setattr("app.core.contrastive_type_knn.load_promoted", _boom)
    monkeypatch.setattr(tac, "_classify_batch", lambda batch: {})
    atoms = [_Atom("Escort access is required at the dock.")]
    tac.classify_atoms(atoms)
    assert called["n"] == 0
    assert atoms[0].atom_type == "scope_item"


def test_the_builder_meta_is_honest(tmp_path, monkeypatch) -> None:
    """The candidate's meta must carry the caveats, the strict tau, and a
    ready bit that follows the measured numbers -- the anti-`gold_acc=1.0`
    shape: a candidate that cannot describe itself cannot be promoted."""
    import json
    import sqlite3
    import sys
    import types

    from app.learning import build_atom_store as bas

    db = tmp_path / "table.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE multitask_rows (task TEXT, text TEXT, label TEXT,"
        " deal_id TEXT, split TEXT, teacher TEXT, source_db TEXT, repr_version INT)"
    )
    rows = []
    for i in range(40):
        rows.append(("atom_type", f"exclusion sentence variant {i} excluded",
                     "exclusion", f"d{i%8}", "train", "llm", "t.db", 0))
    for i in range(10):
        rows.append(("atom_type", f"exclusion holdout variant {i} excluded",
                     "exclusion", f"h{i%3}", "holdout", "llm", "t.db", 0))
    conn.executemany("INSERT INTO multitask_rows VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    class _TinyModel:
        def encode(self, texts, **kw):
            out = []
            for t in texts:
                v = np.zeros(8, dtype=np.float32)
                v[0] = 1.0
                v[1 + (len(t) % 7)] = 0.2
                out.append(v / np.linalg.norm(v))
            return np.array(out)

        def save(self, path):
            from pathlib import Path

            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "stub.json").write_text("{}")

    # sentence_transformers is an optional runtime dependency -- the app imports
    # it lazily inside a function, so its absence is fine there. Patching it by
    # dotted path is not: that needs the module importable, so this test failed
    # with ModuleNotFoundError on any machine without a ~2GB torch install,
    # while testing nothing about embeddings.
    #
    # A stub module keeps the patch target real. What this test actually asserts
    # -- that build_atom_store's meta carries its caveats, tau and ready bit --
    # is unchanged.
    if "sentence_transformers" not in sys.modules:
        stub = types.ModuleType("sentence_transformers")
        stub.SentenceTransformer = lambda *a, **k: _TinyModel()
        monkeypatch.setitem(sys.modules, "sentence_transformers", stub)
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda *a, **k: _TinyModel()
    )
    out = tmp_path / "cand"
    meta = bas.build(table_db=db, out_dir=out)
    assert meta["mode"] == "typed"
    assert meta["tau"] == pytest.approx(0.98)
    assert meta["caveats"], "a candidate without caveats is a 91% waiting to happen"
    assert "repr_versions" in meta
    assert (out / "store.npz").exists() and (out / "knn_meta.json").exists()
    written = json.loads((out / "knn_meta.json").read_text())
    assert written["eval"]["value_light_eligible"] >= 0
