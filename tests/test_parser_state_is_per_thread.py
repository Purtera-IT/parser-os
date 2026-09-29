# -*- coding: utf-8 -*-
"""Per-document parser state must not cross between threads.

``registry._REGISTERED`` holds ONE instance of each parser and hands that same
object to every caller, while ``_prefetch_parses`` runs a thread pool. A parser
that parks per-document state on ``self`` therefore shares it with whatever
else is being parsed at that moment.

Live 010237: the Workato work-order table was given a lead-in -- "The following
response and resolution targets apply to incidents logged through ServiceNow"
-- that belongs to a DIFFERENT document in the same deal. Parsed on its own
that table has no lead-in at all.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.parsers.base import ArtifactParser, PerThreadState
from app.parsers.docx_parser import DocxParser
from app.parsers.registry import get_registered_parsers
from app.parsers.xlsx_parser import XlsxParser


class _Probe(ArtifactParser):
    """Minimal holder; ArtifactParser's abstract methods are never called."""

    value = PerThreadState()

    def match(self, path, sample_text, domain_pack):  # pragma: no cover
        raise NotImplementedError

    def parse(self, *a, **kw):  # pragma: no cover
        raise NotImplementedError

    def parse_artifact_full(self, **kw):  # pragma: no cover
        raise NotImplementedError

    def describe(self):  # pragma: no cover
        raise NotImplementedError


def test_unset_still_raises_so_getattr_defaults_survive() -> None:
    """The parsers read this state as ``getattr(self, name, default)``.

    If an unset attribute returned something instead of raising, every one of
    those defaults would silently stop applying.
    """
    probe = _Probe()
    with pytest.raises(AttributeError):
        probe.value
    assert getattr(probe, "value", "fallback") == "fallback"
    assert getattr(probe, "value", []) == []


def test_one_instance_holds_a_separate_value_per_thread() -> None:
    probe = _Probe()
    probe.value = "main thread"
    seen: dict[str, object] = {}
    start = threading.Barrier(4)

    def worker(n: int) -> None:
        start.wait()
        # Unset for THIS thread even though the main thread set it.
        seen["unset_%d" % n] = getattr(probe, "value", None)
        probe.value = "thread %d" % n
        start.wait()
        seen["after_%d" % n] = probe.value

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))

    for n in range(4):
        assert seen["unset_%d" % n] is None, "state leaked into thread %d" % n
        assert seen["after_%d" % n] == "thread %d" % n, (
            "thread %d read another thread's value: %r" % (n, seen["after_%d" % n])
        )
    assert probe.value == "main thread", "a worker overwrote the main thread"


def test_shared_state_leaks_without_the_descriptor() -> None:
    """The bug this guards against, demonstrated on a plain attribute."""

    class Plain:
        pass

    obj = Plain()
    obj.value = "document A"
    barrier = threading.Barrier(2)

    def other() -> None:
        barrier.wait()
        obj.value = "document B"

    t = threading.Thread(target=other)
    t.start()
    barrier.wait()
    t.join()
    assert obj.value == "document B"  # A's state is simply gone


@pytest.mark.parametrize(
    "parser_cls, attrs",
    [
        (DocxParser, ["_structure_idxs", "_para_lead_in", "_table_lead_in"]),
        (XlsxParser, ["_block_detection_failures", "_coverage_backstop_note",
                      "_sheet_profiles", "_sheet_supplies"]),
    ],
)
def test_per_document_attributes_are_declared_per_thread(parser_cls, attrs) -> None:
    for attr in attrs:
        assert isinstance(getattr(parser_cls, attr, None), PerThreadState), (
            "%s.%s holds per-document state on an instance the registry shares "
            "between threads; declare it PerThreadState()" % (parser_cls.__name__, attr)
        )


def test_registry_really_does_share_one_instance() -> None:
    """The premise. If this ever stops being true the tests above are moot."""
    a = {type(p): id(p) for p in get_registered_parsers()}
    b = {type(p): id(p) for p in get_registered_parsers()}
    assert a == b and a, "registry no longer returns the same parser instances"
