from __future__ import annotations

import base64
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pytest

from app.core.hardware_evidence_backfill import backfill_hardware_bom_lines
from app.core.schemas import AtomType
from app.parsers.email_parser import EmailParser

# 1x1 PNG
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

_OCR_EQUIPMENT_TEXT = "\n".join(
    [
        "Access Point E7 Enterprise × 6",
        "Switch Pro Max 48 PoE × 1",
        "Enterprise NVR × 1",
        "G6 Pro Turret × 4",
        "Card Reader × 3",
    ]
)


def _build_multipart_eml(*, cid: str = "07131976-d75d-4133-b5d2-52a8919274ba") -> bytes:
  mixed = "=_Mixed_test"
  related = "=_Related_test"
  png_b64 = base64.b64encode(_TINY_PNG).decode("ascii")
  lines = [
      "From: patrick@example.com",
      "To: buyer@example.com",
      "Subject: Equipment list",
      "MIME-Version: 1.0",
      f'Content-Type: multipart/mixed; boundary="{mixed}"',
      "",
      f"--{mixed}",
      f'Content-Type: multipart/related; boundary="{related}"',
      "",
      f"--{related}",
      "Content-Type: text/plain; charset=utf-8",
      "",
      f"Full equipment list below.\n[cid:{cid}]\n",
      f"--{related}",
      "Content-Type: text/html; charset=utf-8",
      "",
      f'<p>Full equipment list below.</p><img src="cid:{cid}" />',
      f"--{related}",
      "Content-Type: image/png",
      "Content-Transfer-Encoding: base64",
      f"Content-ID: <{cid}@hubspot-ingest>",
      "",
      png_b64,
      f"--{related}--",
      f"--{mixed}--",
      "",
  ]
  return "\r\n".join(lines).encode("utf-8")


_HUBSPOT_ORDER_TABLE_TEXT = "\n".join(
    [
        "Order Details",
        "Access Card × 10",
        "Protect All-In-One Sensor × 2",
        "Switch Pro Max 48 PoE × 2",
        "Access Point E7 × 6",
        "Enterprise NVR × 1",
        "G6 PTZ Mount × 6",
        "Access G3 Reader × 4",
        "Dream Machine Beast × 2",
        "Camera G6 Pro Turret × 9",
    ]
)


def test_cid_image_ocr_helper_exists() -> None:
    from app.parsers import email_parser as ep

    assert callable(ep._ocr_text_from_cid_image)


def test_transcript_pdf_cid_does_not_emit_spoken_equipment_atoms(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.parsers.email_parser._ocr_text_from_cid_inline",
        lambda _payload, content_type="": (
            "Meeting Summary and Full Transcript\n"
            "Jacob Vander-Plaats [07:16]\n"
            "We have like 4E7 APS. We have two UDM beast for like, their.\n"
        ),
    )
    eml = tmp_path / "transcript-inline.eml"
    eml.write_bytes(_build_multipart_eml())
    atoms = EmailParser().parse_artifact("deal-gecko", "art_transcript_pdf", eml)
    equipment = [a for a in atoms if a.value.get("kind") == "email_cid_equipment_line"]
    assert equipment == []


def test_order_details_table_emits_equipment_atoms(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.parsers.email_parser._ocr_text_from_cid_inline",
        lambda _payload, content_type="": _HUBSPOT_ORDER_TABLE_TEXT,
    )
    eml = tmp_path / "order-table.eml"
    eml.write_bytes(_build_multipart_eml())

    atoms = EmailParser().parse_artifact("deal-gecko", "art_order_table", eml)
    equipment = [a for a in atoms if a.value.get("kind") == "email_cid_equipment_line"]
    assert len(equipment) >= 8


def test_order_details_table_mints_full_bom() -> None:
    class _Scope:
        def __init__(self, text: str, qty: int):
            self.atom_type = type("T", (), {"value": "scope_item"})()
            self.raw_text = text
            self.text = text
            self.value = {"text": text, "kind": "email_cid_equipment_line", "quantity": qty}

    lines = [
        _Scope(line, int(line.rsplit("×", 1)[-1].strip()))
        for line in _HUBSPOT_ORDER_TABLE_TEXT.splitlines()
        if "×" in line
    ]
    out, minted = backfill_hardware_bom_lines(lines, project_id="deal-gecko")
    assert minted >= 8
    bom = {
        a.value["sku"]: a.value["quantity"]
        for a in out
        if getattr(getattr(a, "atom_type", None), "value", "") == "bom_line"
    }
    assert bom.get("UBNT-ACCESS-CARD") == 10
    assert bom.get("UBNT-PROTECT-SENSOR") == 2
    assert bom.get("UBNT-SW-PRO") == 2
    assert bom.get("UBNT-E7-AP") == 6
    assert bom.get("UBNT-NVR") == 1
    assert bom.get("UBNT-G6-PTZ-MOUNT") == 6
    assert bom.get("UBNT-BADGE-READER") == 4
    assert bom.get("UBNT-UDM-BEAST") == 2


def test_multipart_eml_cid_pdf_ocr_emits_equipment_atoms(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.parsers.email_parser._ocr_text_from_cid_inline",
        lambda _payload, content_type="": _OCR_EQUIPMENT_TEXT,
    )
    eml = tmp_path / "inline-equipment-pdf.eml"
    eml.write_bytes(_build_multipart_eml(cid="pdf-cid-1"))
    raw = eml.read_bytes().replace(b"image/png", b"application/pdf")
    eml.write_bytes(raw)

    atoms = EmailParser().parse_artifact("deal-gecko", "art_inline_pdf", eml)
    equipment = [a for a in atoms if a.value.get("kind") == "email_cid_equipment_line"]
    assert equipment

def test_multipart_eml_cid_image_ocr_emits_equipment_atoms(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.parsers.email_parser._ocr_text_from_cid_inline",
        lambda _payload, content_type="": _OCR_EQUIPMENT_TEXT,
    )
    eml = tmp_path / "inline-equipment-img.eml"
    eml.write_bytes(_build_multipart_eml())

    atoms = EmailParser().parse_artifact("deal-gecko", "art_inline_img", eml)
    equipment = [a for a in atoms if a.value.get("kind") == "email_cid_equipment_line"]
    assert equipment


def test_multipart_eml_missing_cid_part_emits_unresolved(tmp_path) -> None:
    eml = tmp_path / "missing-inline.eml"
    eml.write_text(
        "\n".join(
            [
                "From: a@b.com",
                "Subject: test",
                "Content-Type: text/plain; charset=utf-8",
                "",
                "See [cid:missing-image-uuid]",
            ]
        ),
        encoding="utf-8",
    )
    atoms = EmailParser().parse_artifact("deal-gecko", "art_missing", eml)
    unresolved = [a for a in atoms if a.value.get("kind") == "email_cid_unresolved"]
    assert len(unresolved) == 1
    assert "missing-image-uuid" in unresolved[0].value.get("content_ids", [])


def test_hardware_backfill_mints_bom_from_cid_equipment_lines() -> None:
    class _Scope:
        def __init__(self, text: str):
            self.atom_type = type("T", (), {"value": "scope_item"})()
            self.raw_text = text
            self.text = text
            self.value = {"text": text, "kind": "email_cid_equipment_line"}

    atoms = [_Scope(_OCR_EQUIPMENT_TEXT)]
    out, minted = backfill_hardware_bom_lines(atoms, project_id="deal-gecko")
    assert minted >= 4
    bom = {
        a.value["sku"]: a.value["quantity"]
        for a in out
        if getattr(getattr(a, "atom_type", None), "value", "") == "bom_line"
    }
    assert bom.get("UBNT-E7-AP") == 6
    assert bom.get("UBNT-SW-PRO") == 1
    assert bom.get("UBNT-NVR") == 1
    assert bom.get("UBNT-G6-TURRET") == 4 or bom.get("UBNT-G6-PRO-DB") == 4
    assert bom.get("UBNT-BADGE-READER") == 3
    email_bom = [
        a for a in out
        if getattr(getattr(a, "atom_type", None), "value", "") == "bom_line"
        and a.value.get("source") == "email_cid_equipment_line"
    ]
    assert len(email_bom) >= 4


def test_hardware_backfill_maps_order_list_product_names() -> None:
    class _Scope:
        def __init__(self, text: str, qty: int | None = None):
            self.atom_type = type("T", (), {"value": "scope_item"})()
            self.raw_text = text
            self.text = text
            self.value = {"text": text, "kind": "email_cid_equipment_line", "quantity": qty}

    lines = [
        _Scope("Access Point E7 Enterprise × 6", 6),
        _Scope("Switch Pro Max 48 PoE × 1", 1),
        _Scope("Enterprise NVR × 1", 1),
        _Scope("G6 Pro Turret × 4", 4),
        _Scope("Access G3 Reader × 7", 7),
        _Scope("Access Card × 25", 25),
        # HubSpot Order Details screenshot rows (name … qty, no × glyph).
        _Scope("Access Point E7          6"),
        _Scope("Switch Pro Max 48 PoE    2"),
        _Scope("Camera G6 Pro Turret     9"),
        _Scope("Access G3 Reader         4"),
        _Scope("Dream Machine Beast      2"),
        _Scope("Access Card             10"),
    ]
    out, minted = backfill_hardware_bom_lines(lines, project_id="deal-gecko")
    assert minted >= 5
    bom = {
        a.value["sku"]: a.value["quantity"]
        for a in out
        if getattr(getattr(a, "atom_type", None), "value", "") == "bom_line"
    }
    assert bom.get("UBNT-E7-AP") == 6
    assert bom.get("UBNT-SW-PRO") in (1, 2)
    assert bom.get("UBNT-NVR") == 1
    assert bom.get("UBNT-G6-TURRET") in (4, 9)
    assert bom.get("UBNT-BADGE-READER") in (4, 7)
    assert bom.get("UBNT-ACCESS-CARD") in (10, 25)
    assert bom.get("UBNT-UDM-BEAST") == 2


def test_cid_pdf_prefers_digital_text_layer(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fitz

    digital = "\n".join(
        [
            "Order Details",
            "Access Point E7 Enterprise × 6",
            "Switch Pro Max 48 PoE × 1",
            "Enterprise NVR × 1",
        ]
    )

    class _Page:
        def get_text(self, mode="text"):
            if mode == "text":
                return digital
            if mode == "dict":
                return {"blocks": []}
            return ""

        def find_tables(self):
            class _Finder:
                tables = []

            return _Finder()

    class _Doc:
        def __iter__(self):
            yield _Page()

    monkeypatch.setattr("fitz.open", lambda *a, **k: _Doc())
    called = {"ocr": 0}

    def _boom(_page):
        called["ocr"] += 1
        return {"text": "4E7 APS", "backend": "fake", "confidence": 0.1, "notes": []}

    monkeypatch.setattr("app.parsers._ocr_chain.ocr_pdf_page", _boom)
    from app.parsers.email_parser import _ocr_text_from_cid_pdf

    text = _ocr_text_from_cid_pdf(b"%PDF-fake")
    assert "Access Point E7 Enterprise × 6" in text
    assert "Switch Pro Max 48 PoE × 1" in text
    assert called["ocr"] == 0


def test_trailing_order_qty_ignores_switch_model_number() -> None:
    from app.parsers.email_parser import (
        _hardware_atoms_from_equipment_text,
        _sanity_order_qty,
        _trailing_order_qty,
    )

    assert _trailing_order_qty("Switch Pro Max 48 PoE          2") == 2
    assert _sanity_order_qty("Switch Pro Max 48 PoE", 48) is None
    assert _sanity_order_qty("Switch Pro Max 48 PoE", 2) == 2

    text = "\n".join(
        [
            "Order Details",
            "Access Point E7 Enterprise          6",
            "Switch Pro Max 48 PoE               2",
            "Enterprise NVR                      1",
            "Access G3 Reader Pro                4",
            "Access Card                        10",
            "Protect All-In-One Sensor           2",
        ]
    )
    atoms = _hardware_atoms_from_equipment_text(
        project_id="deal-gecko",
        artifact_id="art1",
        filename="e.eml",
        text=text,
        content_id="cid1",
        parser_version="test",
    )
    equipment = [a for a in atoms if a.value.get("kind") == "email_cid_equipment_line"]
    assert len(equipment) >= 5
    by_item = {a.value.get("item", ""): a.value.get("quantity") for a in equipment}
    assert by_item.get("Access Point E7 Enterprise") == 6
    assert by_item.get("Switch Pro Max 48 PoE") == 2
    assert by_item.get("Enterprise NVR") == 1


def test_garbled_ocr_equipment_line_emits_atom(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.parsers.email_parser._ocr_text_from_cid_inline",
        lambda _payload, content_type="": "4E7 APS\nSwitch Pro × 2\nEnterprise NVR × 1",
    )
    eml = tmp_path / "garbled-inline.eml"
    eml.write_bytes(_build_multipart_eml())

    atoms = EmailParser().parse_artifact("deal-gecko", "art_garbled", eml)
    equipment = [a for a in atoms if a.value.get("kind") == "email_cid_equipment_line"]
    assert len(equipment) >= 2
    qty_by_line = {a.raw_text: a.value.get("quantity") for a in equipment}
    assert qty_by_line.get("4E7 APS") == 4


def test_score_cid_ocr_prefers_order_details_over_transcript() -> None:
    from app.parsers.email_parser import _score_cid_ocr_text

    transcript = (
        "Meeting Summary and Full Transcript\n"
        "Jacob Vander-Plaats [07:16]\n"
        "We have like 4E7 APS. We have two UDM beast for like, their.\n"
    )
    order = _HUBSPOT_ORDER_TABLE_TEXT
    assert _score_cid_ocr_text(order) > _score_cid_ocr_text(transcript)


def test_picks_best_cid_when_multiple_inline_images(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript_cid = "transcript-cid-uuid"
    order_cid = "07131976-d75d-4133-b5d2-52a8919274ba"

    def _fake_ocr(payload: bytes, content_type: str = "") -> str:
        if order_cid.encode() in payload or len(payload) < 200:
            return _HUBSPOT_ORDER_TABLE_TEXT
        return (
            "Meeting Summary and Full Transcript\n"
            "Jacob Vander-Plaats [07:16]\n"
            "We have like 4E7 APS.\n"
        )

    monkeypatch.setattr("app.parsers.email_parser._ocr_text_from_cid_inline", _fake_ocr)

    mixed = "=_Mixed_dual"
    related = "=_Related_dual"
    png_b64 = base64.b64encode(_TINY_PNG).decode("ascii")
    lines = [
        "From: patrick@example.com",
        "Subject: Equipment list",
        f'Content-Type: multipart/mixed; boundary="{mixed}"',
        "",
        f"--{mixed}",
        f'Content-Type: multipart/related; boundary="{related}"',
        "",
        f"--{related}",
        "Content-Type: text/plain; charset=utf-8",
        "",
        f"Full equipment list below.\n[cid:{order_cid}]\n",
        f"--{related}",
        "Content-Type: image/png",
        "Content-Transfer-Encoding: base64",
        f"Content-ID: <{transcript_cid}@hubspot-ingest>",
        "",
        png_b64,
        f"--{related}",
        "Content-Type: image/png",
        "Content-Transfer-Encoding: base64",
        f"Content-ID: <{order_cid}@hubspot-ingest>",
        "",
        png_b64 + "extra",
        f"--{related}--",
        f"--{mixed}--",
        "",
    ]
    eml = tmp_path / "dual-inline.eml"
    eml.write_bytes("\r\n".join(lines).encode("utf-8"))

    atoms = EmailParser().parse_artifact("deal-gecko", "art_dual_cid", eml)
    equipment = [a for a in atoms if a.value.get("kind") == "email_cid_equipment_line"]
    assert len(equipment) >= 8
    cids = {a.value.get("content_id") for a in equipment}
    assert order_cid in cids
    assert transcript_cid not in cids


def test_iter_cid_inline_parts_finds_image_part() -> None:
    msg = BytesParser(policy=policy.default).parsebytes(_build_multipart_eml())
    from app.parsers.email_parser import _iter_cid_inline_parts

    parts = _iter_cid_inline_parts(msg)
    assert "07131976-d75d-4133-b5d2-52a8919274ba" in parts
    assert parts["07131976-d75d-4133-b5d2-52a8919274ba"]["is_image"] is True


def test_glued_trailing_qty_on_garbled_switch_and_protect() -> None:
    from app.parsers.email_parser import _hardware_atoms_from_equipment_text

    text = "\n".join(
        [
            "Protect All-In-One Sensor 2",
            "Switch Pro Max 48 PoE 2",
        ]
    )
    atoms = _hardware_atoms_from_equipment_text(
        project_id="deal-gecko",
        artifact_id="art1",
        filename="e.eml",
        text=text,
        content_id="cid-order",
        parser_version="test",
    )
    by_item = {a.value.get("item", ""): a.value.get("quantity") for a in atoms}
    assert by_item.get("Protect All-In-One Sensor") == 2
    assert by_item.get("Switch Pro Max 48 PoE") == 2


def test_equipment_list_ocr_picks_richest_table_over_transcript(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript_cid = "07131976-d75d-4133-b5d2-52a8919274ba"
    order_cid = "f41c1a3b-2993-42e3-a181-e2441b3942d0"
    transcript_ocr = "\n".join(
        [
            "Meeting Summary and Full Transcript",
            "Protect All-In-One Sensor 2",
            "Switch Pro Max 48 PoE 2",
        ]
    )

    def _fake_ocr_part(part: dict) -> str:
        cid = str(part.get("content_id") or "")
        if cid == order_cid:
            return _HUBSPOT_ORDER_TABLE_TEXT
        if cid == transcript_cid:
            return transcript_ocr
        return ""

    monkeypatch.setattr("app.parsers.email_parser._ocr_cid_part", _fake_ocr_part)

    mixed = "=_Mixed_gecko"
    related = "=_Related_gecko"
    png_b64 = base64.b64encode(_TINY_PNG).decode("ascii")
    body = "Below is the full equipment list.\n"
    lines = [
        "From: patrick@example.com",
        "Subject: Equipment list",
        f'Content-Type: multipart/mixed; boundary="{mixed}"',
        "",
        f"--{mixed}",
        f'Content-Type: multipart/related; boundary="{related}"',
        "",
        f"--{related}",
        "Content-Type: text/plain; charset=utf-8",
        "",
        body + f"[cid:{transcript_cid}]\n[cid:{order_cid}]\n",
        f"--{related}",
        "Content-Type: image/png",
        "Content-Transfer-Encoding: base64",
        f"Content-ID: <{transcript_cid}@hubspot-ingest>",
        "",
        png_b64,
        f"--{related}",
        "Content-Type: image/png",
        "Content-Transfer-Encoding: base64",
        f"Content-ID: <{order_cid}@hubspot-ingest>",
        "",
        png_b64 + "x" * 120,
        f"--{related}--",
        f"--{mixed}--",
        "",
    ]
    eml = tmp_path / "gecko-dual-inline.eml"
    eml.write_bytes("\r\n".join(lines).encode("utf-8"))

    atoms = EmailParser().parse_artifact("deal-gecko", "art_gecko_dual", eml)
    equipment = [a for a in atoms if a.value.get("kind") == "email_cid_equipment_line"]
    unresolved = [a for a in atoms if a.value.get("kind") == "email_cid_unresolved"]
    assert len(equipment) >= 8
    assert {a.value.get("content_id") for a in equipment} == {order_cid}
    assert unresolved == []
