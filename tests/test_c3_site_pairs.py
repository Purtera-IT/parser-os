"""The site-pair head (jdg:site) is asked of the type head's site lines and
served under ``site_pairs`` (sites_audit.md: it was never requested)."""
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from ml.c3 import serve  # noqa: E402
from ml.c3.predict import _judge_site_pairs  # noqa: E402


class _Model:
    def __init__(self):
        self.asked = []

    def judge(self, out, key, subjects, queries=None):
        self.asked.append((key, list(subjects)))
        return torch.tensor([[0.0, 0.0, 3.0]] * len(subjects))  # distinct_site


def _schema():
    opp = NS(answers=[NS(value="_absent"), NS(value="same_site"), NS(value="distinct_site")])
    return NS(by_key=lambda: {"jdg:site": opp})


def _lines(types):
    return {k: {"reads": {"col:label_type": {"answer": t, "p": 0.9}}} for k, t in types.items()}


def test_every_pair_of_site_lines_is_asked_once():
    model, pairs, judged = _Model(), [], set()
    batch = NS(keys=["a", "b", "c", "d"])
    lines = _lines({"a": "physical_site", "b": "task", "c": "physical_site", "d": "physical_site"})
    _judge_site_pairs(model, None, batch, lines, _schema(), pairs, judged)
    assert model.asked == [("jdg:site", [(0, 2), (0, 3), (2, 3)])]
    assert [p[:3] for p in pairs] == [["a", "c", "distinct_site"], ["a", "d", "distinct_site"],
                                      ["c", "d", "distinct_site"]]
    _judge_site_pairs(model, None, batch, lines, _schema(), pairs, judged)   # a later window holding the same lines
    assert len(model.asked) == 1 and len(pairs) == 3


def test_the_service_returns_the_pairs_only_when_asked():
    seen = {}

    def read(model, deal, schema, **kw):
        seen.update(kw)
        return {"lines": {}, "site_pairs": [["a", "c", "distinct_site", 0.9]]}

    base = serve.Base(load=lambda: ("m", object(), "cpu"), read=read)
    base.load()
    payload = {"deal_id": "d", "company": "purtera", "atoms": []}
    assert "site_pairs" not in base.read(payload, 10.0)
    assert "site_pairs" not in seen
    out = base.read(payload, 10.0, site_pairs=True)
    assert seen["site_pairs"] is True and out["site_pairs"] == [["a", "c", "distinct_site", 0.9]]
