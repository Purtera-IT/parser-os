"""Parsing 68 documents one at a time is why big deals died.

`parse_artifacts` took 881 of a 1500-second budget on a 68-document deal --
13 seconds apiece, almost none of it computing. Each document is a blob
download, an OCR round trip and a vision call allowed 90 seconds; the process
spends that time holding a socket. The loop ran them strictly in series, so
the deal hit the wall before the stages that USE the atoms ever ran.

The prefetch warms those calls concurrently. What it must not do is change a
single thing about the result -- above all the ORDER. Atom ordering feeds
`label_key`, so a completion-ordered parse would re-key every atom on every
compile and silently detach every label ever written.
"""
from __future__ import annotations

import app.core.compiler as C


class _Result:
    def __init__(self, name):
        self.atoms = [f"{name}-atom"]
        self.candidates = []
        self.warnings = []
        self.derived_files = []


class _Parser:
    """Returns after a delay that REVERSES the input order if completion order
    ever leaked through: the first artifact is slowest."""
    def __init__(self, delay):
        self.delay = delay

    def parse_artifact_full(self, *, project_id, artifact_id, path, domain_pack):
        import time
        time.sleep(self.delay)
        return _Result(artifact_id)


def _plan(n):
    # a[0] sleeps longest, a[n-1] shortest
    return [(f"a{i}", _Parser(delay=(n - i) * 0.02), f"/p/{i}") for i in range(n)]


def test_every_artifact_is_parsed():
    out = C._prefetch_parses(_plan(6), project_id="p", domain_pack=None, workers=4)
    assert set(out) == {f"a{i}" for i in range(6)}
    assert out["a3"].atoms == ["a3-atom"]


def test_results_are_keyed_not_ordered():
    """The prefetch returns a MAPPING, so the caller consumes it in the
    artifacts' own order. Completion order cannot leak into atom order because
    completion order is never what is iterated."""
    out = C._prefetch_parses(_plan(6), project_id="p", domain_pack=None, workers=4)
    assert isinstance(out, dict)
    ordered = [out[f"a{i}"].atoms[0] for i in range(6)]
    assert ordered == [f"a{i}-atom" for i in range(6)]


def test_one_failure_does_not_take_the_others():
    """A parser that raises is simply absent from the map, and the compile
    loop parses it inline exactly as it did before."""
    class _Boom(_Parser):
        def parse_artifact_full(self, **kw):
            raise RuntimeError("bad file")

    plan = _plan(4)
    plan[1] = ("a1", _Boom(0), "/p/1")
    out = C._prefetch_parses(plan, project_id="p", domain_pack=None, workers=4)
    assert "a1" not in out
    assert {"a0", "a2", "a3"} <= set(out)


def test_workers_of_one_disables_it():
    """An escape hatch that restores the exact previous behaviour."""
    assert C._prefetch_parses(_plan(4), project_id="p", domain_pack=None, workers=1) == {}
    assert C._prefetch_parses(_plan(4), project_id="p", domain_pack=None, workers=0) == {}


def test_a_single_artifact_is_not_worth_a_pool():
    assert C._prefetch_parses(_plan(1), project_id="p", domain_pack=None, workers=8) == {}


def test_it_is_actually_concurrent():
    """Six artifacts sleeping 0.12s..0.02s is 0.42s in series. In a pool of 6
    it is the longest one, 0.12s. Generous bound so a slow machine passes."""
    import time

    start = time.time()
    C._prefetch_parses(_plan(6), project_id="p", domain_pack=None, workers=6)
    assert time.time() - start < 0.30
