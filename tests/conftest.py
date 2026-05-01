from __future__ import annotations

from pathlib import Path

import pytest

from scripts.make_demo_fixtures import create_demo_project


@pytest.fixture(autouse=True)
def _reset_active_domain_pack() -> None:
    """Avoid cross-test leakage via mutable domain-pack singleton state."""
    from app.domain import load_domain_pack, set_active_domain_pack

    set_active_domain_pack(load_domain_pack(None))
    yield
    set_active_domain_pack(load_domain_pack(None))


@pytest.fixture()
def demo_project(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    return create_demo_project(root)
