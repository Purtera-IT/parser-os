"""The parser service never imports the C3 model package (ml/c3), and ml/c3
never imports the service. parser-os is bundled into parser-os-worker, so an
import from app/ into ml/ would ship torch and the model into production
parsing. This check needs no torch."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_never_imports_ml():
    pat = re.compile(r"^\s*(from|import)\s+ml(\.|\s|$)", re.M)
    hits = [str(p.relative_to(ROOT)) for p in (ROOT / "app").rglob("*.py")
            if pat.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert hits == []


def test_ml_never_imports_app():
    pat = re.compile(r"^\s*(from|import)\s+app(\.|\s|$)", re.M)
    hits = [str(p.relative_to(ROOT)) for p in (ROOT / "ml").rglob("*.py")
            if pat.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert hits == []


def test_ml_is_not_packaged_with_the_parser():
    toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    include = re.search(r"include\s*=\s*\[([^\]]*)\]", toml).group(1)
    assert "ml" not in re.findall(r'"([^"*]+)\*?"', include)
