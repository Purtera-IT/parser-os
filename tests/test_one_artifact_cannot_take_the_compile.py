"""One artifact must not be able to take the whole compile.

Deal 5bd32822 carried an SSRS customer report — `SSRS-SL-CUS001-CustomerOut`,
15,253 rows x 43 columns, one sheet — that parsed to **94,047 atoms** (48,321 of
them `scope_item`) and serialised to a **319 MB** cache payload. The compile ran
6.8 hours, twice, and blocked every deploy behind it, because the drain will not
roll over a running compile.

Measured on the file itself: parse 28s, model_dump 2.4s, json.dumps 0.9s. So
nothing hangs — the cost is everything downstream having to carry 94,000 atoms
of somebody's customer master data.

Dropped, not truncated. Keeping the first N atoms of a 94,000-atom file is a
silent, arbitrary sample of one report presented as this deal's evidence. A
skip says so, and the routing row records what was discarded so a genuinely
oversized scope file is visible rather than mysterious.
"""

from __future__ import annotations

import inspect

from app.core import compiler


def test_there_is_a_cap_at_all() -> None:
    assert compiler._MAX_ATOMS_PER_ARTIFACT > 0
    # Comfortably above any real deal document, well under the 94,047 that
    # prompted it.
    assert 1000 < compiler._MAX_ATOMS_PER_ARTIFACT < 94047


def test_it_is_tunable_without_a_deploy() -> None:
    src = inspect.getsource(compiler)
    assert 'os.environ.get("SOWSMITH_MAX_ATOMS_PER_ARTIFACT"' in src


def test_the_oversized_artifact_is_skipped_not_truncated() -> None:
    """Truncation would present an arbitrary slice of a customer report as this
    deal's evidence, which is worse than not parsing it: it is wrong AND
    invisible."""
    src = inspect.getsource(compiler.compile_project)
    body = src[src.index("_MAX_ATOMS_PER_ARTIFACT and len(parsed_atoms)"):]
    head = body[: body.index("continue")]
    assert "skipped_oversized" in head
    assert "atoms_discarded" in head, "say how much was dropped, or nobody can tell"
    # No slicing of the atom list anywhere in the guard.
    assert "parsed_atoms[:" not in head


def test_the_skip_is_loud() -> None:
    src = inspect.getsource(compiler.compile_project)
    guard = src[src.index("_MAX_ATOMS_PER_ARTIFACT and len(parsed_atoms)"):]
    guard = guard[: guard.index("continue")]
    assert "parse_warnings.append" in guard, "a silent skip is how a file vanishes"
    assert "SOWSMITH_MAX_ATOMS_PER_ARTIFACT" in guard, "tell the reader how to override it"


def test_a_normal_artifact_is_untouched() -> None:
    """The cap must not change the ordinary path — the 9-artifact Yealink deal
    produced 567 atoms across every file."""
    assert 567 < compiler._MAX_ATOMS_PER_ARTIFACT
    assert 94047 > compiler._MAX_ATOMS_PER_ARTIFACT
