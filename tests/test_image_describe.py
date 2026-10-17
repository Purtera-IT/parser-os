"""Image facts: one line per fact the vision model reports, nothing for logos."""
from app.core import image_describe as I


def test_facts_come_back_as_lines(monkeypatch):
    from app.core import pdf_image_vision

    monkeypatch.setattr(pdf_image_vision, "_vlm", lambda b, p, model, max_tokens:
                        '{"meaningful": true, "facts": ["3 Aruba APs are marked.", "Outdoor brick wall."]}')
    assert I.describe(b"x" * 20000, "plan.docx", "Red Hexagons are Aruba APs") == [
        "3 Aruba APs are marked.", "Outdoor brick wall."]


def test_a_logo_or_a_tiny_image_gives_nothing(monkeypatch):
    from app.core import pdf_image_vision

    monkeypatch.setattr(pdf_image_vision, "_vlm", lambda b, p, model, max_tokens: '{"meaningful": false, "facts": []}')
    assert I.describe(b"x" * 20000, "logo.png") == []
    assert I.describe(b"x" * 100, "tiny.png") == []


def test_no_legend_no_counted_marks_and_no_absence_claims(monkeypatch):
    from app.core import pdf_image_vision

    monkeypatch.setattr(pdf_image_vision, "_vlm", lambda b, p, model, max_tokens:
                        '{"meaningful": true, "facts": ["There are 12 devices marked with a black square.", '
                        '"No outdoor mounting surfaces are shown.", "The wall is outdoor brick."]}')
    assert I.describe(b"x" * 20000, "plan.png") == ["The wall is outdoor brick."]
    got = I.describe(b"x" * 20000, "plan.docx", "Red Hexagons are Aruba APs")
    assert "There are 12 devices marked with a black square." in got
