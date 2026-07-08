from __future__ import annotations

from app.core.manifest_artifact_dedup import dedupe_manifest_email_artifacts


def test_dedupe_prefers_multipart_hs_email_over_plain_text_stub() -> None:
    stale = {
        "attachment_id": "4fd9784b-06c8-4e27-82ab-127b8b90dd5f",
        "filename": "010058-hs-email-111652731176.eml",
        "content_sha256": "f0edf641ece19ea23fb8374eb42221c460022cb5108092f926e2bb055fd4768b",
        "size_bytes": 2186,
        "mime_type": "message/rfc822",
        "source": "email",
    }
    fresh = {
        "attachment_id": "97840ef6-76bd-46e1-b75d-355e0c60f8c0",
        "filename": "010058-hs-email-111652731176.eml",
        "content_sha256": "b5171b2e5b0f11e03da75e48ab7d9f633cd070192e69c787ac596f56352b5847",
        "size_bytes": 41998,
        "mime_type": "message/rfc822",
        "source": "email",
        "external_id": "hs-email:111652731176",
        "metadata": {
            "emlBuilderVersion": "2",
            "hasBodyHtml": True,
            "inlineImageParts": 1,
        },
    }
    note = {
        "attachment_id": "note-1",
        "filename": "010058-hs-note-112019851881-note.txt",
        "external_id": "hs-note:112019851881",
        "size_bytes": 152,
    }

    out = dedupe_manifest_email_artifacts([stale, note, fresh])
    emails = [a for a in out if "hs-email" in str(a.get("filename") or "")]
    assert len(emails) == 1
    assert emails[0]["content_sha256"] == fresh["content_sha256"]
    assert any(a.get("attachment_id") == "note-1" for a in out)


def test_dedupe_matches_hs_email_id_from_filename_when_external_id_missing() -> None:
    a = {
        "filename": "deal-hs-email-42.eml",
        "size_bytes": 1500,
    }
    b = {
        "filename": "deal-hs-email-42.eml",
        "size_bytes": 9000,
        "metadata": {"emlBuilderVersion": "2", "inlineImageParts": 1},
    }
    out = dedupe_manifest_email_artifacts([a, b])
    assert len(out) == 1
    assert out[0]["size_bytes"] == 9000
