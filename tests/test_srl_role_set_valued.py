from types import SimpleNamespace

from app.core.orbitbrief_core import _role_text


def test_a_set_valued_role_reads_as_text():
    assert _role_text(["Project_Manager", "site_contact"]) == "project_manager site_contact"
    assert "project_manager" in _role_text(["project_manager"])


def test_a_plain_or_missing_role_still_reads():
    assert _role_text("Facilities Manager") == "facilities manager"
    assert _role_text(None) == ""
    assert SimpleNamespace(value={"role": None}).value.get("role") is None
