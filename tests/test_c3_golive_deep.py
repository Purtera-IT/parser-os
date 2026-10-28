"""Go-live, deep-audit batch: C3 failure is loud, an empty compile fails and is never priced, the
parser-yield warning matches versions, the prefetch and quote stages end, the schematic pre-pass
and the lessons lookups are off. Hermetic; all text is made up."""
from __future__ import annotations

import io
import urllib.error
from pathlib import Path
from types import SimpleNamespace as N

import pytest

import app.core.c3_type_base as c3
from app.core import telemetry
from app.core.compiler import compile_project

URL = "http://heads.invalid"


def _http(code):
    return urllib.error.HTTPError(URL + "/read", code, f"HTTP {code}", {}, io.BytesIO(b""))


@pytest.fixture()
def heads_500(monkeypatch):
    for k in ("SOWSMITH_C3_FALLBACK", "SOWSMITH_C3_QUOTE", "SOWSMITH_C3_BRIEF"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SOWSMITH_C3_URL", URL)
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "on")
    monkeypatch.setenv("SOWSMITH_C3_AUTHORITY", "on")

    def _req(path, body, timeout):
        if path == "/health":
            return {"ready": True}
        raise _http(500)

    monkeypatch.setattr(c3, "_request", _req)
    monkeypatch.setattr(c3, "warm", lambda: None)


# ------------------------------------------------------------- (1) loud C3 failure

def test_a_default_path_read_error_is_recorded(heads_500):
    from tests.test_c3_type_base import _atoms

    c3.set_deal([])
    assert c3.classify(_atoms()) == {}
    got = {(f["where"], f["state"]) for f in c3.failures()}
    assert got == {("doc-kind read", "read_failed"), ("type read", "read_failed")}
    c3.set_deal([])
    assert c3.failures() == []


def test_a_compile_whose_heads_fail_is_degraded_and_says_so(heads_500, tmp_path):
    d = tmp_path / "deal_c3_down"
    d.mkdir()
    (d / "notes.md").write_text("Install four access points in the lobby.\nThe customer supplies the mounts.\n")
    result = compile_project(project_dir=d, use_cache=False, allow_errors=True, allow_unverified_receipts=True)
    assert any(w.startswith("WARNING: C3 heads unavailable") and "HTTP 500" in w for w in result.warnings)
    assert result.compile_capabilities["compile_state"] == "degraded"
    assert result.compile_capabilities["c3_ok"] is False


# ------------------------------------------------------------- (2) zero evidence

def test_an_empty_compile_is_failed_and_never_priced(monkeypatch, tmp_path):
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", URL)
    monkeypatch.setattr(c3, "_request", lambda *a, **k: pytest.fail("nothing may be priced"))
    d = tmp_path / "deal_empty"
    d.mkdir()
    result = compile_project(project_dir=d, use_cache=False)
    assert result.compile_capabilities["compile_state"] == "failed"
    assert any("0 evidence atoms" in w for w in result.warnings)
    assert c3.quote([N(id="a", raw_text="  ", atom_type="task")], {"sections": {}}) == {}


# ------------------------------------------------------------- (3) parser yield

def test_parser_yield_matches_the_coded_version(tmp_path):
    d = tmp_path / "deal_yield"
    d.mkdir()
    (d / "scope.md").write_text("Install four access points in the lobby.\nThe customer supplies the mounts.\n")
    result = compile_project(project_dir=d, use_cache=False, allow_errors=True, allow_unverified_receipts=True)
    routing = result.manifest.parser_routing
    assert routing and "+code" in str(routing[0].get("parser_version"))  # the suffix that broke the match
    assert result.atoms
    assert result.quality.parsers_with_zero_atoms == []
    assert not any("produced 0 atoms" in w for w in result.warnings)


# ------------------------------------------------------------- (4) stages end

def test_prefetch_is_a_timed_stage_and_quote_ends_its_stage():
    src = Path("app/core/compiler.py").read_text()
    pre = src.index('telemetry.stage("parse_prefetch"')
    assert src.index("telemetry.end_stage(_pstage", pre) < src.index('telemetry.stage("parse_artifacts"')
    q = src.index('telemetry.stage("c3_quote"')
    assert "telemetry.end_stage(_qstage" in src[q:q + 1200]


def test_prefetch_ticks_progress(monkeypatch):
    from app.core import compiler

    ticks = []
    monkeypatch.setattr(telemetry, "set_stage_progress", lambda d, t: ticks.append((d, t)))
    parser = N(parse_artifact_full=lambda **k: k["artifact_id"])
    got = compiler._prefetch_parses([("a1", parser, "x"), ("a2", parser, "y")], project_id="p",
                                    domain_pack=None, workers=2)
    assert got == {"a1": "a1", "a2": "a2"} and sorted(ticks) == [(1, 2), (2, 2)]


# ------------------------------------------------------------- (5) schematic pre-pass off

def test_the_schematic_prepass_is_off_by_default(monkeypatch):
    from app.parsers import orbitbrief_pdf

    monkeypatch.delenv("SOWSMITH_SCHEMATIC_PREPASS", raising=False)
    assert orbitbrief_pdf.schematic_prepass_enabled() is False
    monkeypatch.setenv("SOWSMITH_SCHEMATIC_PREPASS", "1")
    assert orbitbrief_pdf.schematic_prepass_enabled() is True


def test_a_pdf_parse_never_calls_the_prepass_when_off(monkeypatch, tmp_path):
    import fitz

    from app.parsers import orbitbrief_pdf

    monkeypatch.delenv("SOWSMITH_SCHEMATIC_PREPASS", raising=False)
    monkeypatch.setattr(orbitbrief_pdf, "_run_schematic_pre_pass",
                        lambda **k: pytest.fail("the retired schematic pre-pass ran"))
    p = tmp_path / "plan.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Install four access points in the lobby.")
    doc.save(p)
    d = tmp_path / "deal_pdf"
    d.mkdir()
    (d / "plan.pdf").write_bytes(p.read_bytes())
    result = compile_project(project_dir=d, use_cache=False, allow_errors=True, allow_unverified_receipts=True)
    assert not any("schematic" in str(getattr(a.atom_type, "value", a.atom_type)) for a in result.atoms)


# ------------------------------------------------------------- (6) lessons lookup off

def test_decide_skips_the_store_when_lookups_are_off(monkeypatch):
    from app.core import decide as dec

    monkeypatch.delenv("SOWSMITH_LESSON_LOOKUP", raising=False)
    store = N(resolve=lambda **k: pytest.fail("the lessons store was consulted"), get_deal_context=None)
    monkeypatch.setattr(dec, "_STORE", store)
    d = dec.decide("atom_type", "Install four access points.", ["task", "scope_item"], instruction="type", llm=False)
    assert d.source != "store"


def test_no_lesson_warmup_when_lookups_are_off(monkeypatch, tmp_path):
    import app.core.embedding_retrieval as er

    monkeypatch.delenv("SOWSMITH_LESSON_LOOKUP", raising=False)
    calls = []
    monkeypatch.setattr(er, "embed_texts", lambda texts, *a, **k: calls.append(len(texts)) or [])
    d = tmp_path / "deal_lessons"
    d.mkdir()
    (d / "notes.md").write_text("Install four access points in the lobby.\n")
    result = compile_project(project_dir=d, use_cache=False, allow_errors=True, allow_unverified_receipts=True)
    assert not any("lesson-lookup embeddings warmed" in w for w in result.warnings)
