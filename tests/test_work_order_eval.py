"""Corpus measurement for work_order (PUR-31 repro spread, PUR-46 off-vs-on diff).

Synthetic only: a fake compile reads the flag and returns hand-built atoms, so
these run without an LLM, a database or any real deal.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from app.core import work_order_eval as ev
from app.core.schemas import AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def _atom(text, artifact_id="art-1", *, minted=False, ref_artifact=None, atom_id=None):
    ref_art = ref_artifact or artifact_id
    return EvidenceAtom(
        id=atom_id or f"a-{abs(hash((text, artifact_id, minted))) % 10**9}",
        project_id="p1",
        artifact_id=artifact_id,
        atom_type=AtomType.scope_item,
        raw_text=text,
        normalized_text=text.lower(),
        authority_class=AuthorityClass.customer_current_authored,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        parser_version="1",
        source_refs=[
            SourceRef(
                id=f"sr-{ref_art}", artifact_id=ref_art, artifact_type="email",
                filename="scope.eml", locator={"line": 1},
                extraction_method="email_parser", parser_version="1",
            )
        ],
        value={"backfill_reason": "work_order"} if minted else {},
    )


BASE = [_atom("Install 11 access points"), _atom("After 6pm only"), _atom("Partnership chat", "art-2")]


def _corpus(tmp_path: Path, names=("deal_a", "deal_b")) -> Path:
    for n in names:
        (tmp_path / n).mkdir()
        (tmp_path / n / "scope.txt").write_text("synthetic")
    (tmp_path / "empty_dir").mkdir()
    (tmp_path / ".hidden").mkdir()
    return tmp_path


def test_list_deals_filters(tmp_path):
    c = _corpus(tmp_path, ("b", "a", "marion"))
    assert [p.name for p in ev.list_deals(c)] == ["a", "b", "marion"]
    assert [p.name for p in ev.list_deals(c, exclude=["marion"])] == ["a", "b"]
    assert [p.name for p in ev.list_deals(c, include=["b"])] == ["b"]


def test_flag_env_restores():
    os.environ.pop(ev.FLAG, None)
    with ev.flag_env(True):
        assert os.environ[ev.FLAG] == "1"
    assert ev.FLAG not in os.environ
    os.environ[ev.FLAG] = "yes"
    with ev.flag_env(False):
        from app.core import work_order
        assert not work_order.enabled()
    assert os.environ.pop(ev.FLAG) == "yes"


def test_soft_agreement_ignores_paraphrase():
    run1 = ["Update documentation", "Confirm connectivity", "Test systems and troubleshoot"]
    run2 = ["Update Access One documentation", "Test systems with client and troubleshoot"]
    exact = ev.work_order.work_line_agreement([run1, run2])
    soft = ev.soft_work_line_agreement([run1, run2])
    assert exact == 0.0
    assert soft == 2 / 3
    assert ev.soft_work_line_agreement([["a"], ["a"]]) == 1.0
    assert ev.soft_work_line_agreement([[], []]) == 1.0


def test_repro_and_summary(tmp_path):
    corpus = _corpus(tmp_path)
    calls = {"n": 0}

    def fake_compile(deal):
        calls["n"] += 1
        extra = [] if deal.name == "deal_a" or calls["n"] % 2 else [_atom("Confirm connectivity", minted=True)]
        return SimpleNamespace(atoms=BASE + [_atom("Install access points", minted=True)] + extra)

    rows = ev.run_repro(ev.list_deals(corpus), fake_compile, runs=4)
    by = {r["deal_id"]: r for r in rows}
    assert by["deal_a"]["agreement"] == 1.0 and by["deal_a"]["line_count_spread"] == 0
    assert by["deal_b"]["agreement"] < 1.0 and by["deal_b"]["line_counts"] in ([1, 2, 1, 2], [2, 1, 2, 1])
    s = ev.summarize_repro(rows, threshold=0.9)
    assert s["deals"] == 2 and s["runs_per_deal"] == 4
    assert s["deals_below_threshold"] == ["deal_b"]
    assert s["min_agreement"] == by["deal_b"]["agreement"]


def test_diff_additive_deal():
    minted = _atom("install access points", minted=True)
    row = ev.diff_deal("d", list(BASE), [BASE + [minted], BASE + [minted]], 1.0, [2.0, 2.2])
    assert row["atoms_off"] == 3 and row["atoms_on"] == 4 and row["atom_delta"] == 1
    assert row["work_lines_added"] == ["install access points"]
    assert row["atoms_lost_count"] == 0 and row["provenance_problems"] == []
    assert row["agreement"]["agreement"] == 1.0
    assert "lost_atoms" not in row["flags"] and "bad_provenance" not in row["flags"]
    assert "big_atom_delta" in row["flags"]  # 1/3 > 25%


def test_diff_flags_lost_atoms_bad_provenance_and_collapse():
    many = [_atom(f"line {i}", atom_id=f"x{i}") for i in range(ev.COLLAPSE_MIN_ATOMS)]
    ghost = _atom("do the job", "art-nowhere", minted=True)
    on = many[1:] + [ghost]
    row = ev.diff_deal("barton", many, [on], 1.0, [1.0])
    assert row["atoms_lost_count"] == 1
    assert row["provenance_problems"][0]["unknown_artifacts"] == ["art-nowhere"]
    assert {"lost_atoms", "bad_provenance", "collapse"} <= set(row["flags"])
    assert row["agreement"] is None


def test_run_corpus_diff_uses_flag_and_survives_errors(tmp_path):
    corpus = _corpus(tmp_path, ("good", "broken"))
    seen = []

    def fake_compile(deal):
        on = ev.work_order.enabled()
        seen.append((deal.name, on))
        if deal.name == "broken" and on:
            raise RuntimeError("boom")
        return SimpleNamespace(atoms=BASE + ([_atom("install", minted=True)] if on else []))

    got = []
    rows = ev.run_corpus_diff(ev.list_deals(corpus), fake_compile, on_runs=2, on_row=got.append)
    assert [r["deal_id"] for r in rows] == ["broken", "good"] == [r["deal_id"] for r in got]
    assert ("good", False) in seen and seen.count(("good", True)) == 2
    assert rows[0]["flags"] == ["error"]
    s = ev.summarize_diff(rows)
    assert s["errors"] == 1 and s["additive"] and s["work_lines_total"] == 1
    md = ev.render_markdown(rows, s)
    assert "| good | 3 | 4 | +1 | 1 | 0 | 0 | 1.0 |" in md and "error: RuntimeError: boom" in md
    json.dumps(rows)


def test_corpus_diff_cli_writes_reports(tmp_path, monkeypatch):
    import scripts.work_order_corpus_diff as cli
    import app.core.compiler as compiler

    corpus = _corpus(tmp_path / "c" if (tmp_path / "c").mkdir() is None else tmp_path)
    monkeypatch.setattr(
        compiler, "compile_project",
        lambda deal, **k: SimpleNamespace(atoms=BASE + ([_atom("install", minted=True)] if ev.work_order.enabled() else [])),
    )
    out = tmp_path / "out"
    assert cli.main([str(corpus), "--out", str(out), "--exclude", "deal_b"]) == 0
    assert json.loads((out / "summary.json").read_text())["deals"] == 1
    assert (out / "deals" / "deal_a.json").exists() and (out / "report.md").exists()
    assert len((out / "rows.jsonl").read_text().splitlines()) == 1
