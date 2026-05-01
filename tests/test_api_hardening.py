from __future__ import annotations
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from scripts.make_demo_fixtures import create_demo_project


def _create_project(client: TestClient) -> str:
    response = client.post("/projects", json={"name": "Hardening"})
    assert response.status_code == 200
    return response.json()["project_id"]


def _upload(client: TestClient, project_id: str, filename: str, content: bytes) -> int:
    response = client.post(
        f"/projects/{project_id}/artifacts",
        files={"file": (filename, content, "application/octet-stream")},
    )
    return response.status_code


def test_health_live_works() -> None:
    with TestClient(app) as client:
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_version_works() -> None:
    with TestClient(app) as client:
        response = client.get("/version")
        assert response.status_code == 200
        payload = response.json()
        assert "schema_version" in payload
        assert "compiler_version" in payload
        assert "packetizer_version" in payload
        assert "authority_policy_version" in payload


def test_bad_extension_rejected() -> None:
    with TestClient(app) as client:
        project_id = _create_project(client)
        response = client.post(
            f"/projects/{project_id}/artifacts",
            files={"file": ("bad.exe", b"abc", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Unsupported file extension" in response.json()["detail"]


def test_path_traversal_filename_rejected() -> None:
    with TestClient(app) as client:
        project_id = _create_project(client)
        response = client.post(
            f"/projects/{project_id}/artifacts",
            files={"file": ("../evil.txt", b"abc", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Path traversal" in response.json()["detail"]


def test_empty_file_rejected() -> None:
    with TestClient(app) as client:
        project_id = _create_project(client)
        response = client.post(
            f"/projects/{project_id}/artifacts",
            files={"file": ("empty.txt", b"", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()


def test_large_file_rejected_with_small_config(monkeypatch) -> None:
    monkeypatch.setenv("PURTERA_MAX_UPLOAD_BYTES", "16")
    with TestClient(app) as client:
        project_id = _create_project(client)
        response = client.post(
            f"/projects/{project_id}/artifacts",
            files={"file": ("large.txt", b"0123456789abcdefghij", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "max size" in response.json()["detail"]
    monkeypatch.delenv("PURTERA_MAX_UPLOAD_BYTES", raising=False)


def test_packet_pagination_works(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir(parents=True, exist_ok=True)
    demo_project = create_demo_project(fixture_root)
    with TestClient(app) as client:
        project_id = _create_project(client)
        for artifact in sorted(demo_project.iterdir()):
            if artifact.is_file():
                with artifact.open("rb") as fh:
                    response = client.post(
                        f"/projects/{project_id}/artifacts",
                        files={"file": (artifact.name, fh, "application/octet-stream")},
                    )
                assert response.status_code == 200
        compile_response = client.post(f"/projects/{project_id}/compile")
        assert compile_response.status_code == 200

        page_one = client.get(f"/projects/{project_id}/packets", params={"limit": 2, "offset": 0})
        assert page_one.status_code == 200
        payload = page_one.json()
        assert payload["limit"] == 2
        assert payload["offset"] == 0
        assert len(payload["items"]) <= 2
        assert payload["total"] >= len(payload["items"])


def test_packet_filtering_by_family_status_severity(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir(parents=True, exist_ok=True)
    demo_project = create_demo_project(fixture_root)
    with TestClient(app) as client:
        project_id = _create_project(client)
        for artifact in sorted(demo_project.iterdir()):
            if artifact.is_file():
                with artifact.open("rb") as fh:
                    response = client.post(
                        f"/projects/{project_id}/artifacts",
                        files={"file": (artifact.name, fh, "application/octet-stream")},
                    )
                assert response.status_code == 200
        assert client.post(f"/projects/{project_id}/compile").status_code == 200
        response = client.get(
            f"/projects/{project_id}/packets",
            params={"family": "vendor_mismatch", "status": "needs_review", "severity": "high"},
        )
        assert response.status_code == 200
        items = response.json()["items"]
        assert all(item["family"] == "vendor_mismatch" for item in items)
        assert all(item["status"] == "needs_review" for item in items)
        assert all(item["risk"]["severity"] == "high" for item in items)


def test_atom_filtering_by_authority_class(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir(parents=True, exist_ok=True)
    demo_project = create_demo_project(fixture_root)
    with TestClient(app) as client:
        project_id = _create_project(client)
        for artifact in sorted(demo_project.iterdir()):
            if artifact.is_file():
                with artifact.open("rb") as fh:
                    response = client.post(
                        f"/projects/{project_id}/artifacts",
                        files={"file": (artifact.name, fh, "application/octet-stream")},
                    )
                assert response.status_code == 200
        assert client.post(f"/projects/{project_id}/compile").status_code == 200
        response = client.get(
            f"/projects/{project_id}/atoms",
            params={"authority_class": "vendor_quote"},
        )
        assert response.status_code == 200
        items = response.json()["items"]
        assert items
        assert all(item["authority_class"] == "vendor_quote" for item in items)
