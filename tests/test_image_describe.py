"""Image facts: one line per fact the vision model reports, nothing for logos."""
from app.core import image_describe as I


def test_facts_come_back_as_lines(monkeypatch):
    from app.core import pdf_image_vision

    monkeypatch.setattr(pdf_image_vision, "_vlm", lambda b, p, model, max_tokens:
                        '{"meaningful": true, "facts": ["3 Aruba APs are marked.", "Outdoor brick wall."]}')
    assert I.describe(b"x" * 20000, "plan.docx") == ["3 Aruba APs are marked.", "Outdoor brick wall."]


def test_a_logo_or_a_tiny_image_gives_nothing(monkeypatch):
    from app.core import pdf_image_vision

    monkeypatch.setattr(pdf_image_vision, "_vlm", lambda b, p, model, max_tokens: '{"meaningful": false, "facts": []}')
    assert I.describe(b"x" * 20000, "logo.png") == []
    assert I.describe(b"x" * 100, "tiny.png") == []
