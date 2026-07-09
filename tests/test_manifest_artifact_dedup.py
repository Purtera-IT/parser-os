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


def test_upgrade_stub_hs_email_to_multipart_sibling() -> None:
    from app.core.manifest_artifact_dedup import upgrade_stub_hs_email_artifacts_from_siblings

    stub = {
        "attachment_id": "4fd9784b-06c8-4e27-82ab-127b8b90dd5f",
        "filename": "010058-hs-email-111652731176.eml",
        "content_sha256": "f0edf641ece19ea23fb8374eb42221c460022cb5108092f926e2bb055fd4768b",
        "size_bytes": 2186,
        "mime_type": "message/rfc822",
        "source": "email",
        "blob_url": (
            "https://purpulsedevstg01.blob.core.windows.net/orbitbrief-artifacts/"
            "deals/c2912e57/artifacts/f0edf641ece19ea23fb8374eb42221c460022cb5108092f926e2bb055fd4768b/"
            "010058-hs-email-111652731176.eml"
        ),
    }
    note = {
        "attachment_id": "note-1",
        "filename": "010058-hs-note-1.txt",
        "size_bytes": 100,
        "blob_url": "https://example/blob",
    }
    siblings = [
        (
            "deals/c2912e57/artifacts/f0edf641ece19ea23fb8374eb42221c460022cb5108092f926e2bb055fd4768b/"
            "010058-hs-email-111652731176.eml",
            2186,
        ),
        (
            "deals/c2912e57/artifacts/cea177032186d1fc2bc2a4126d0f1850e83317aaba20979ba68fa6ecb9f2015a/"
            "010058-hs-email-111652731176.eml",
            1004926,
        ),
        (
            "deals/c2912e57/artifacts/cea177032186d1fc2bc2a4126d0f1850e83317aaba20979ba68fa6ecb9f2015a/"
            "other.txt",
            50,
        ),
    ]

    def _list(_container: str, _prefix: str) -> list[tuple[str, int]]:
        assert _prefix == "deals/c2912e57/artifacts/"
        return siblings

    out = upgrade_stub_hs_email_artifacts_from_siblings(
        [stub, note],
        list_blobs=_list,
        account_host="purpulsedevstg01.blob.core.windows.net",
    )
    emails = [a for a in out if "hs-email" in str(a.get("filename") or "")]
    assert len(emails) == 1
    assert emails[0]["size_bytes"] == 1004926
    assert "cea17703" in emails[0]["blob_url"]
    assert emails[0]["metadata"].get("upgradedFromStub") is True
    assert emails[0]["external_id"] == "hs-email:111652731176"
    assert any(a.get("attachment_id") == "note-1" for a in out)


def test_upgrade_leaves_multipart_hs_email_alone() -> None:
    from app.core.manifest_artifact_dedup import upgrade_stub_hs_email_artifacts_from_siblings

    fresh = {
        "filename": "010058-hs-email-111652731176.eml",
        "size_bytes": 1004926,
        "blob_url": (
            "https://purpulsedevstg01.blob.core.windows.net/orbitbrief-artifacts/"
            "deals/c2912e57/artifacts/cea17703/010058-hs-email-111652731176.eml"
        ),
        "metadata": {"emlBuilderVersion": "3", "inlineImageParts": 2},
    }

    def _list(_c: str, _p: str) -> list[tuple[str, int]]:
        raise AssertionError("should not list siblings for non-stub")

    out = upgrade_stub_hs_email_artifacts_from_siblings([fresh], list_blobs=_list)
    assert out[0]["size_bytes"] == 1004926
    assert out[0].get("metadata", {}).get("upgradedFromStub") is None
