"""An Outlook .msg whose body is HTML-only must not lose its body.

Audit 2026-10-28 (wave 12, W3): an RFQ .msg had ``body`` None and the whole
request in ``htmlBody``; the parser emitted the header atom and nothing else.
Fixture text is synthetic.
"""
from __future__ import annotations

import sys
import types


def _run(monkeypatch, tmp_path, *, body, html):
    class _FakeMsg:
        subject = "Clinic installs"
        sender = "Pat Sender <pat@vendor.example>"
        date = "Mon, 05 Oct 2026 10:00:00 -0500"
        attachments: list = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    _FakeMsg.body = body
    _FakeMsg.htmlBody = html
    monkeypatch.setitem(sys.modules, "extract_msg", types.SimpleNamespace(openMsg=lambda _p: _FakeMsg()))
    from app.parsers._universal_extras import MsgParser

    target = tmp_path / "rfq.msg"
    target.write_bytes(b"\xd0\xcf\x11\xe0stub")
    out = MsgParser().parse_artifact_full(project_id="p", artifact_id="a", path=target)
    return out.atoms


_HTML = (
    b"<html><head><style>p.MsoNormal{margin:0in}</style></head><body>"
    b"<p class=MsoNormal>Hi team,</p>"
    b"<p>We need smart hands at about forty clinics over the next year.</p>"
    b"<ul><li>Rack and stack two switches per clinic.</li>"
    b"<li>Label and dress every patch cable.</li></ul>"
    b"<table><tr><td>Overtime</td><td>1.5x</td></tr></table>"
    b"</body></html>"
)


def test_html_only_msg_body_is_read(monkeypatch, tmp_path):
    atoms = _run(monkeypatch, tmp_path, body=None, html=_HTML)
    texts = [a.raw_text for a in atoms]
    joined = "\n".join(texts)
    assert "smart hands at about forty clinics" in joined
    assert "Rack and stack two switches per clinic." in texts
    assert "Label and dress every patch cable." in texts
    assert "MsoNormal" not in joined, "CSS from <head> is not the sender's text"
    assert any("Overtime" in t and "1.5x" in t for t in texts), "a table row stays one line"
    body = [a for a in atoms if (a.source_refs[0].locator or {}).get("kind") == "msg_body"]
    assert body and all(a.source_refs[0].locator.get("body_part") == "html" for a in body)


def test_plain_body_still_wins_over_html(monkeypatch, tmp_path):
    atoms = _run(monkeypatch, tmp_path, body="Install the kiosk on Tuesday.", html=_HTML)
    texts = [a.raw_text for a in atoms]
    assert "Install the kiosk on Tuesday." in texts
    assert not any("forty clinics" in t for t in texts)
