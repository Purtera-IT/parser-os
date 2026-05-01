from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from scripts.make_demo_fixtures import create_demo_project


def test_api_smoke_end_to_end(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture_root"
    fixture_root.mkdir(parents=True, exist_ok=True)
    demo_project = create_demo_project(fixture_root)

    with TestClient(app) as client:
        create_resp = client.post("/projects", json={"name": "Demo"})
        assert create_resp.status_code == 200
        project_id = create_resp.json()["project_id"]

        for artifact in sorted(demo_project.iterdir()):
            if artifact.is_dir():
                continue
            with artifact.open("rb") as fh:
                upload_resp = client.post(
                    f"/projects/{project_id}/artifacts",
                    files={"file": (artifact.name, fh, "application/octet-stream")},
                )
            assert upload_resp.status_code == 200
            payload = upload_resp.json()
            assert payload["project_id"] == project_id

        compile_resp = client.post(f"/projects/{project_id}/compile")
        assert compile_resp.status_code == 200
        compile_json = compile_resp.json()
        assert compile_json["project_id"] == project_id

        packets_resp = client.get(f"/projects/{project_id}/packets")
        assert packets_resp.status_code == 200
        packets_payload = packets_resp.json()
        assert {"total", "limit", "offset", "items"} <= set(packets_payload.keys())
        packets = packets_payload["items"]
        families = {packet["family"] for packet in packets}
        assert "quantity_conflict" in families
        assert "scope_exclusion" in families

        filtered_family = client.get(f"/projects/{project_id}/packets", params={"family": "quantity_conflict"})
        assert filtered_family.status_code == 200
        family_payload = filtered_family.json()
        assert family_payload["items"]
        assert all(packet["family"] == "quantity_conflict" for packet in family_payload["items"])

        filtered_status = client.get(f"/projects/{project_id}/packets", params={"status": "needs_review"})
        assert filtered_status.status_code == 200
        status_payload = filtered_status.json()
        assert all(packet["status"] == "needs_review" for packet in status_payload["items"])

        atoms_resp = client.get(f"/projects/{project_id}/atoms")
        edges_resp = client.get(f"/projects/{project_id}/edges")
        entities_resp = client.get(f"/projects/{project_id}/entities")
        assert atoms_resp.status_code == 200
        assert edges_resp.status_code == 200
        assert entities_resp.status_code == 200
        assert {"total", "limit", "offset", "items"} <= set(atoms_resp.json().keys())
        assert {"total", "limit", "offset", "items"} <= set(edges_resp.json().keys())
        assert {"total", "limit", "offset", "items"} <= set(entities_resp.json().keys())
