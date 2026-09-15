"""Router scope summary v3: the deal's own name leads, and conversation lines give way
to written scope when there is enough of it (live 010162)."""
from types import SimpleNamespace

from app.core.service_router import SCOPE_SUMMARY_VERSION, _CAP, _scope_summary


def _a(atom_type, text):
    return SimpleNamespace(atom_type=atom_type, raw_text=text)


def test_deal_name_leads_the_summary():
    atoms = [_a("scope_item", f"Deploy SD-WAN appliances at site {i}") for i in range(6)]
    out = _scope_summary(atoms, [{"filename": "recap.txt"}], deal_name="CDW- Sodexo SD-WAN Program")
    assert out.startswith("DEAL: CDW- Sodexo SD-WAN Program\nFILES: recap")
    assert SCOPE_SUMMARY_VERSION == 3


def test_no_deal_line_without_a_name():
    atoms = [_a("scope_item", f"Install display {i}") for i in range(6)]
    assert _scope_summary(atoms, []).startswith("FILES:")


def test_utterances_give_way_to_written_scope():
    written = [_a("scope_item", f"Fixed price per SD WAN deployment site, item {i}") for i in range(_CAP // 2)]
    spoken = [_a("raw_utterance", t) for t in ("How you doing, buddy?", "Amen.", "Okay.")]
    out = _scope_summary(spoken + written, [])
    assert "Amen." not in out and "How you doing" not in out
    assert "SD WAN deployment site" in out


def test_utterances_stay_when_written_scope_is_thin():
    written = [_a("scope_item", "Install one display")] * 3
    spoken = [_a("raw_utterance", f"We need a technician on site day {i}") for i in range(4)]
    out = _scope_summary(spoken + written, [])
    assert "technician on site" in out
