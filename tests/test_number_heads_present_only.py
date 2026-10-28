"""A number head's value is served only on lines its own answer says carry one (010095 / 000051 kit holdouts)."""
from ml.c3.predict import gate_numbers


def test_a_number_on_an_absent_line_is_not_served():
    reads = [{"read:co_hours_estimate": {"answer": "_absent", "p": 0.97, "value": 4336.268}},
             {"read:co_hours_estimate": {"answer": "present", "p": 0.9, "value": 6.0}},
             {"read:labor_hours": {"value": 12.0}}]  # no answer head: left as it is
    out = gate_numbers(reads)
    assert "value" not in out[0]["read:co_hours_estimate"] and out[0]["read:co_hours_estimate"]["value_absent"] == 4336.268
    assert out[1]["read:co_hours_estimate"]["value"] == 6.0
    assert out[2]["read:labor_hours"]["value"] == 12.0
