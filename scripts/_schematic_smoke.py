"""Lightweight smoke runner for the schematic upgrade tests.

Pytest hits STATUS_STACK_BUFFER_OVERRUN on this Windows/CPython 3.12.3
dev environment (see pyproject.toml note). This runner imports a test
module and invokes every top-level ``test_*`` function so the schematic
PRs can be validated locally without pytest.
"""
from __future__ import annotations

import importlib
import inspect
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _materialize_fixture(name: str, fn) -> object:
    if name == "tmp_path":
        import tempfile

        p = Path(tempfile.mkdtemp(prefix="schematic_smoke_"))
        return p
    raise SystemExit(f"smoke runner does not handle fixture: {name}")


def _parametrize_cases(fn) -> list[tuple[str, dict]]:
    """Expand pytest.mark.parametrize markers into (label, kwargs) pairs.

    Supports two forms:
      - single argname ``"x"`` with iterable of values
      - comma-separated argnames ``"a,b"`` with iterable of value tuples
    Stacked parametrize markers (one per arg group) are AND'd together.
    """
    base: list[tuple[str, dict]] = [("", {})]
    marks = getattr(fn, "pytestmark", [])
    for mark in marks:
        if getattr(mark, "name", "") != "parametrize":
            continue
        args = getattr(mark, "args", None) or ()
        if len(args) < 2:
            continue
        argname, values = args[0], args[1]
        if not isinstance(argname, str):
            continue
        argnames = [n.strip() for n in argname.split(",") if n.strip()]
        expansions: list[tuple[str, dict]] = []
        for value in values:
            if len(argnames) == 1:
                value_tuple = (value,)
            else:
                value_tuple = tuple(value)
            if len(value_tuple) != len(argnames):
                continue
            kwargs = dict(zip(argnames, value_tuple))
            label = ",".join(f"{k}={kwargs[k]!r}" for k in argnames)
            expansions.append((label, kwargs))
        new_base: list[tuple[str, dict]] = []
        for prefix_label, prefix_kwargs in base:
            for label, kwargs in expansions:
                merged = dict(prefix_kwargs)
                merged.update(kwargs)
                full_label = f"{prefix_label}|{label}" if prefix_label else label
                new_base.append((full_label, merged))
        base = new_base
    if base == [("", {})]:
        return []
    return base


def run_module(modname: str) -> int:
    mod = importlib.import_module(modname)
    failures = 0
    tests = sorted([n for n in dir(mod) if n.startswith("test_")])
    for name in tests:
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        param_cases = _parametrize_cases(fn)
        if param_cases:
            for label, extra in param_cases:
                kwargs: dict[str, object] = dict(extra)
                for pname in sig.parameters:
                    if pname in kwargs:
                        continue
                    kwargs[pname] = _materialize_fixture(pname, fn)
                try:
                    fn(**kwargs)
                    print(f"  PASS  {modname}::{name}[{label}]")
                except Exception:
                    print(f"  FAIL  {modname}::{name}[{label}]")
                    traceback.print_exc()
                    failures += 1
            continue
        kwargs = {}
        for pname in sig.parameters:
            kwargs[pname] = _materialize_fixture(pname, fn)
        try:
            fn(**kwargs)
            print(f"  PASS  {modname}::{name}")
        except Exception:
            print(f"  FAIL  {modname}::{name}")
            traceback.print_exc()
            failures += 1
    return failures


def main(argv: list[str]) -> int:
    if not argv:
        argv = [
            "tests.test_schematic_contracts",
            "tests.test_pdf_bbox_replay",
        ]
    total = 0
    for modname in argv:
        print(f"=== {modname} ===")
        total += run_module(modname)
    print(f"\nfailures: {total}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
