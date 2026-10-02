"""A deal's own documents are never set aside over a number in their filename (live 010353).

Deal 010353 "VC Links" (Ox, compile b40e9bb3, 2026-10-02): its SOW and Deal
Kit arrived as "010329-OX-0036-VC Links.pdf" and "010328-OX-0035-VC Links.xlsx"
-- Ox deals reuse earlier OX numbers and request ids -- and document_job_scope
set aside all 284 of their atoms as another job, with reasons [] in the
envelope. Both files say "VC Links, 15733 US-224 Findlay".
"""
from types import SimpleNamespace

import pytest

from app.core import decide as decide_mod
from app.core import semantic_role
from app.core.decide import Decision
from app.core.document_job_scope import judge_documents, strip_refs, verdict_note

DEAL = "010353-OX-0038-VC Links"


def _atom(doc, filename, text, n):
    return SimpleNamespace(id=f"{doc}_{n}", source_artifact_id=doc, source_filename=filename, raw_text=text,
                           value={}, review_flags=[], atom_type="scope_item")


SOW = [
    _atom("art_sow", "010329-OX-0036-VC Links.pdf", "Statement of Work - VC Links", 0),
    _atom("art_sow", "010329-OX-0036-VC Links.pdf", "Site: 15733 US-224, Findlay, OH", 1),
    _atom("art_sow", "010329-OX-0036-VC Links.pdf", "Install and terminate 24 Cat6 drops in the warehouse", 2),
]
KIT = [
    _atom("art_kit", "010328-OX-0035-VC Links.xlsx", "Deal Kit | VC Links | 15733 US-224 Findlay", 0),
    _atom("art_kit", "010328-OX-0035-VC Links.xlsx", "Cat6 drop | 24 | 185.00", 1),
]
OTHER = [
    _atom("art_other", "Kiosk close-down.pdf", "Pack and ship 14 kiosks from the Delta Admin building", 0),
]


@pytest.fixture
def no_store():
    prev = decide_mod.get_store()
    decide_mod.set_store(None)
    yield
    decide_mod.set_store(prev)


def test_strip_refs_keeps_the_words_and_drops_the_filing_numbers():
    assert strip_refs("010329-OX-0036-VC Links.pdf") == "VC Links.pdf"
    assert strip_refs("010353-OX-0038-VC Links") == "VC Links"
    assert strip_refs("010162 - CDW- Sodexo SD-WAN Program") == "CDW- Sodexo SD-WAN Program"
    assert strip_refs("POS Installation 8/2") == "POS Installation 8/2"


def test_a_judge_that_reads_numbers_never_sees_them(no_store, monkeypatch):
    # A judge that compares the deal's number with the file's is exactly the
    # failure: the deal's number must not reach it.
    seen = []

    def judge(text, cands, **k):
        seen.append(text + "\n" + str(k.get("context") or ""))
        # Calls it another job when the deal line's number differs from the file's.
        ctx = str(k.get("context") or "")
        return ("other_job", 0.95) if ("010353" in ctx or "OX-0038" in ctx) else ("this_deal", 0.9)

    monkeypatch.setattr(semantic_role, "classify_role", judge)
    # Content that does not name the deal, so only the judge decides.
    bare = [_atom("art_b", "010329-OX-0036-VC Links.pdf", "Install and terminate 24 Cat6 drops", 0)]
    kept, dropped, verdicts = judge_documents(bare, deal_name=DEAL, project_id="010353")
    assert not dropped and len(kept) == 1
    assert all(v["verdict"] == "this_deal" and not v["content_spared"] for v in verdicts)
    assert seen and all("DEAL: VC Links\n" in s and "010353" not in s and "OX-0038" not in s for s in seen)


def test_content_naming_the_deal_keeps_it_whatever_the_judge_says(no_store, monkeypatch):
    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: ("other_job", 0.99))
    kept, dropped, verdicts = judge_documents(SOW + KIT + OTHER, deal_name=DEAL, project_id="010353")
    assert {a.source_artifact_id for a in kept} == {"art_sow", "art_kit"}
    assert [a.source_artifact_id for a in dropped] == ["art_other"]
    spared = {v["filename"]: v for v in verdicts if v["content_spared"]}
    assert set(spared) == {"010329-OX-0036-VC Links.pdf", "010328-OX-0035-VC Links.xlsx"}
    assert all(v["content_match"] == "vc links" for v in spared.values())
    assert "names this deal" in verdict_note(spared["010329-OX-0036-VC Links.pdf"])


def test_a_lesson_on_a_look_alike_title_does_not_remove_the_deals_own_files(no_store, monkeypatch):
    class _Store:
        def resolve(self, *, relation, text, candidates, **_):
            if relation == "document_job" and "VC Links" in text:
                return Decision(verdict="other_job", confidence=0.95, source="store", correction_id="corr_old")
            return None

    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: (None, 0.0))
    decide_mod.set_store(_Store())
    kept, dropped, _ = judge_documents(SOW + KIT, deal_name=DEAL, project_id="010353")
    assert not dropped and len(kept) == 5


def test_the_site_address_from_the_crm_is_an_anchor_too(no_store, monkeypatch):
    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: ("other_job", 0.99))
    site_only = [_atom("art_s", "010329-OX-0036.pdf", "Work at 15733 US-224 Findlay OH 45840", 0)]
    kept, dropped, _ = judge_documents(site_only, deal_name="010353 - Fiber Run", project_id="010353",
                                       crm={"site_address": "15733 US-224 Findlay"})
    assert not dropped and len(kept) == 1


def test_every_drop_carries_a_reason_and_its_atoms(no_store, monkeypatch):
    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: ("other_job", 0.97))
    _, dropped, verdicts = judge_documents(OTHER, deal_name=DEAL, project_id="010353")
    assert len(dropped) == 1
    (v,) = verdicts
    assert v["verdict"] == "other_job"
    assert v["reason"] and "other_job 0.97" in v["reason"] and "VC Links" in v["reason"]
    assert v["atom_ids"] == ["art_other_0"]
    assert v["reason"] in verdict_note(v)


# ── end to end: the envelope and the ledger say why ─────────────────────────


def _deal_project(tmp_path):
    import json

    project = tmp_path / "deal"
    project.mkdir()
    (project / "010329-OX-0036-VC Links.txt").write_text(
        "Statement of Work for VC Links.\n\n"
        "Site address: 15733 US-224, Findlay, OH.\n\n"
        "Purtera will install and terminate 24 Cat6 drops in the warehouse.\n", encoding="utf-8")
    (project / "Kiosk close-down.txt").write_text(
        "Statement of Work for the Delta Admin kiosk close-down.\n\n"
        "Site address: 1030 Delta Blvd, Atlanta, GA.\n\n"
        "Purtera will remove, pack and ship 14 kiosks back to the depot.\n", encoding="utf-8")
    manifest = {"context": {"crm": {"deal_name": "010353-OX-0038-VC Links"}}, "artifacts": []}
    (project / ".parser_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return project


def test_compile_keeps_the_deals_file_and_says_why_the_other_went(no_store, monkeypatch, tmp_path):
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import build_orbitbrief_envelope

    monkeypatch.setenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", "1")
    real = semantic_role.classify_role

    def judge(text, cands, **k):
        if cands == ["this_deal", "other_job"]:
            return ("other_job", 0.97)
        return real(text, cands, **k)

    monkeypatch.setattr(semantic_role, "classify_role", judge)
    project = _deal_project(tmp_path)
    result = compile_project(project_dir=project, project_id="010353", use_cache=False)
    envelope = build_orbitbrief_envelope(project_dir=project, compile_result=result)
    docs = {d["filename"]: d for d in envelope["documents"]}

    sow = docs["010329-OX-0036-VC Links.txt"]["scope"]
    assert sow.get("atoms_admitted", 0) > 0 and not sow.get("atoms_suppressed")

    other = docs["Kiosk close-down.txt"]["scope"]
    assert other["atoms_suppressed"] > 0 and other["suppressed_by_stage"].get("document_job_scope")
    assert other["reasons"], "a set-aside document must say why"
    assert any("document_job_scope" in r and "other_job 0.97" in r for r in other["reasons"])

    rows = [r for r in envelope["suppressed"] if r["stage"] == "document_job_scope"]
    assert rows and all("other_job 0.97" in r["reason"] for r in rows)
    ledger = [a for a in result.suppressed_atoms if "suppressed:document_job_scope" in (a.review_flags or [])]
    assert ledger and all("other_job 0.97" in a.value["_suppression"]["reason"] for a in ledger)
