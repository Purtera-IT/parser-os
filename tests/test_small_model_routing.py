"""Small structured LLM jobs go to SOWSMITH_SMALL_MODEL by stage; the brief does not."""
from app.core import llm_client, telemetry


def test_small_model_by_stage(monkeypatch):
    monkeypatch.setenv("SOWSMITH_SMALL_MODEL", "gpt-4.1-nano")
    monkeypatch.setattr(telemetry, "CURRENT_STAGE", "typed_atom_classification")
    assert llm_client._small_model_for_stage() == "gpt-4.1-nano"
    monkeypatch.setattr(telemetry, "CURRENT_STAGE", "c3_brief")
    assert llm_client._small_model_for_stage() is None
    monkeypatch.delenv("SOWSMITH_SMALL_MODEL")
    monkeypatch.setattr(telemetry, "CURRENT_STAGE", "bom_owner")
    assert llm_client._small_model_for_stage() is None
