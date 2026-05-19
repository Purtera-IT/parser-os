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

    Only supports the simple single-argname form used by the schematic
    test suite. Multi-arg/indirect parametrize is not handled — those
    tests should be exercised via real pytest.
    """
    cases: list[tuple[str, dict]] = []
    marks = getattr(fn, "pytestmark", [])
    for mark in marks:
        if getattr(mark, "name", "") != "parametrize":
            continue
        args = getattr(mark, "args", None) or ()
        if len(args) < 2:
            continue
        argname, values = args[0], args[1]
        if isinstance(argname, str) and "," not in argname:
            for value in values:
                cases.append((f"{argname}={value!r}", {argname: value}))
    return cases


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
