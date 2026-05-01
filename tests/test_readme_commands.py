from __future__ import annotations

from pathlib import Path


def test_readme_contains_required_sections_and_commands() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "Evidence Compiler: artifacts -> atoms -> graph -> packets" in readme
    assert "no chatbot" in readme
    assert "no OrbitBrief" in readme
    assert "no SOW generation" in readme
    assert "no dispatch" in readme
    assert "no VisionQC" in readme
    assert "## Quickstart" in readme
    assert "bash scripts/demo_compile.sh" in readme
    assert "## Architecture" in readme
    assert "## Data Contracts" in readme
    assert "## Quality Gates" in readme
    assert "## Next Milestone" in readme
    assert "OrbitBrief v0 consumes packets and creates scope truth board." in readme
