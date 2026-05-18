"""Minimal pytest-free runner for the takeoff tests.

The pytest 8/9 binaries on this Windows demo environment trigger a
STATUS_STACK_BUFFER_OVERRUN (0xC0000409) at collection time, with no
recoverable output. This runner imports each test module directly and
invokes every top-level ``test_*`` function so we can verify the
takeoff layer without relying on the pytest harness.

It is intentionally NOT installed as a CLI entry point — this is a
development aid, not a replacement for pytest.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

# Make tests/ importable as a flat dir (this script is meant to live
# alongside pytest, not replace its discovery — so we add tests/ to
# sys.path and import each test file as a top-level module).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = _REPO_ROOT / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _resolve_param_id(value: Any) -> str:
    return repr(value)


def _expand_parametrize(func: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    """Return [(case_id, kwargs), ...] for a function carrying pytest.mark.parametrize.

    Supports the most common form: a single ``@pytest.mark.parametrize(
    "a, b, c", [(...), (...), ...])`` decoration. Unsupported forms
    return a single empty-kwargs case so the function is still executed
    (without parametrization).
    """
    pytestmark = getattr(func, "pytestmark", None)
    if not pytestmark:
        return [("", {})]
    for mark in pytestmark:
        if getattr(mark, "name", "") != "parametrize":
            continue
        try:
            arg_spec, value_set = mark.args[0], mark.args[1]
        except Exception:
            continue
        names = [n.strip() for n in arg_spec.split(",")] if isinstance(arg_spec, str) else list(arg_spec)
        cases: list[tuple[str, dict[str, Any]]] = []
        for entry in value_set:
            if not isinstance(entry, (tuple, list)):
                entry = (entry,)
            kwargs = dict(zip(names, entry))
            case_id = ",".join(_resolve_param_id(kwargs[n]) for n in names)
            cases.append((case_id, kwargs))
        return cases
    return [("", {})]


def _maybe_call_skipif(func: Any) -> str | None:
    """Return a skip reason if a skipif mark would trip, else None."""
    pytestmark = getattr(func, "pytestmark", None)
    if not pytestmark:
        return None
    for mark in pytestmark:
        if getattr(mark, "name", "") != "skipif":
            continue
        cond = mark.args[0] if mark.args else False
        reason = mark.kwargs.get("reason", "skipif")
        if cond:
            return str(reason)
    return None


def run_module(module_name: str) -> tuple[int, int, int, list[str]]:
    """Execute every ``test_*`` function in ``module_name``.

    Returns ``(passed, failed, skipped, failure_messages)``.
    """
    module = importlib.import_module(module_name)
    test_funcs = [
        (name, value)
        for name, value in vars(module).items()
        if name.startswith("test_") and inspect.isfunction(value)
    ]
    passed = failed = skipped = 0
    messages: list[str] = []

    # Module-scoped fixtures (very minimal: only resolves the
    # marriott_takeoff fixture used in test_takeoff_marriott_wn).
    fixture_cache: dict[str, Any] = {}

    import tempfile

    def _resolve_kwargs(func: Any) -> dict[str, Any]:
        sig = inspect.signature(func)
        kwargs: dict[str, Any] = {}
        for param in sig.parameters.values():
            if param.name in fixture_cache:
                kwargs[param.name] = fixture_cache[param.name]
                continue
            # Built-in: pytest tmp_path fixture stand-in.
            if param.name in {"tmp_path", "tmpdir"}:
                tmp = Path(tempfile.mkdtemp(prefix="takeoff_test_"))
                kwargs[param.name] = tmp
                continue
            fixture = getattr(module, param.name, None)
            if fixture is not None and getattr(fixture, "__wrapped__", None):
                resolved = fixture.__wrapped__()
                fixture_cache[param.name] = resolved
                kwargs[param.name] = resolved
            elif fixture is not None:
                # Plain function fixture — call it.
                try:
                    resolved = fixture()
                except Exception as exc:
                    raise RuntimeError(
                        f"could not resolve fixture {param.name}"
                    ) from exc
                fixture_cache[param.name] = resolved
                kwargs[param.name] = resolved
        return kwargs

    for name, func in test_funcs:
        skip_reason = _maybe_call_skipif(func)
        if skip_reason:
            skipped += 1
            continue
        for case_id, params in _expand_parametrize(func):
            label = f"{module_name}::{name}" + (f"[{case_id}]" if case_id else "")
            try:
                kwargs = _resolve_kwargs(func)
                kwargs.update(params)
                func(**kwargs)
                passed += 1
                print(f"PASS {label}")
            except _ExpectedSkip as e:
                skipped += 1
                print(f"SKIP {label} :: {e}")
            except Exception as exc:
                # Treat pytest.skip.Exception as a skip.
                exc_name = type(exc).__name__
                if exc_name in {"Skipped"}:
                    skipped += 1
                    print(f"SKIP {label} :: {exc}")
                    continue
                failed += 1
                msg = traceback.format_exc()
                messages.append(f"FAIL {label}\n{msg}")
                print(f"FAIL {label}")
                print(msg)

    return passed, failed, skipped, messages


class _ExpectedSkip(Exception):
    """Internal sentinel — not actually raised in v0."""


def main(argv: list[str]) -> int:
    modules = argv[1:] or [
        # Universal layer — page classification, extraction, zoning.
        "test_takeoff_sheet_classifier",
        "test_takeoff_page_type_router",
        "test_takeoff_multipliers",
        "test_takeoff_pdf_native",
        "test_takeoff_zones",
        "test_takeoff_spatial_zones",
        "test_takeoff_nearby_text",
        "test_takeoff_keynotes",
        "test_takeoff_shape_signals",
        "test_takeoff_ocr_signals",
        "test_takeoff_universal_patterns",
        "test_takeoff_typical_plan_expander",
        # Knowledge stack — reference + per-page intelligence.
        "test_takeoff_legend_self_extractor",
        "test_takeoff_project_reference",
        "test_takeoff_parser_intelligence",
        "test_takeoff_qa_overlay",
        # Pricing layer.
        "test_takeoff_quote_unitizer",
        # End-to-end invariant — WN=335 on the Marriott corpus.
        "test_takeoff_marriott_wn",
    ]
    total_pass = total_fail = total_skip = 0
    all_messages: list[str] = []
    for mod in modules:
        passed, failed, skipped, messages = run_module(mod)
        total_pass += passed
        total_fail += failed
        total_skip += skipped
        all_messages.extend(messages)
    print()
    print(f"=== {total_pass} passed, {total_fail} failed, {total_skip} skipped ===")
    if all_messages:
        print()
        for m in all_messages:
            print(m)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
