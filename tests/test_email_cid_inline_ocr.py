from __future__ import annotations

import base64
from email import policy
from email.parser import BytesParser

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


def test_multipart_eml_cid_image_ocr_emits_equipment_atoms(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.parsers.email_parser._ocr_text_from_cid_image",
        lambda _payload: _OCR_EQUIPMENT_TEXT,
    )
    eml = tmp_path / "inline-equipment.eml"
    eml.write_bytes(_build_multipart_eml())

    atoms = EmailParser().parse_artifact("deal-gecko", "art_inline", eml)
    equipment = [
        a
        for a in atoms
        if a.value.get("kind") == "email_cid_equipment_line"
    ]
    assert equipment, "expected email_cid_equipment_line atoms from OCR text"
    assert any("e7" in (a.value.get("item") or "").lower() for a in equipment)


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
    assert bom.get("UBNT-G6-PRO-DB") == 4
    assert bom.get("UBNT-BADGE-READER") == 3


def test_iter_cid_inline_parts_finds_image_part() -> None:
    msg = BytesParser(policy=policy.default).parsebytes(_build_multipart_eml())
    from app.parsers.email_parser import _iter_cid_inline_parts

    parts = _iter_cid_inline_parts(msg)
    assert "07131976-d75d-4133-b5d2-52a8919274ba" in parts
    assert parts["07131976-d75d-4133-b5d2-52a8919274ba"]["is_image"] is True
