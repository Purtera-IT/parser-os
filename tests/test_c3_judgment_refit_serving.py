"""A judgment_fit add-on (e.g. jdg:document_job) corrects the live document judgment."""
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from ml.c3.judgment_fit import JudgmentRefit, load_refits  # noqa: E402
from ml.c3.predict import _judge_documents  # noqa: E402


class _Model:
    def __init__(self, refits=None):
        if refits is not None:
            self.judgment_refits = refits

    def judge(self, out, key, subjects, queries=None):
        return torch.tensor([[0.0, 2.0, 0.0]] * len(subjects))  # this_deal

    def judgment_features(self, out, key, subjects, queries):
        return torch.ones(len(subjects), 4)


def _schema():
    opp = NS(answers=[NS(value="_absent"), NS(value="this_deal"), NS(value="other_job")])
    return NS(by_key=lambda: {"jdg:document_job": opp})


def _run(model):
    batch = NS(keys=["a1", "a2"])
    atoms = {"a1": NS(doc_id="d1"), "a2": NS(doc_id="d1")}
    docs = {}
    _judge_documents(model, None, batch, atoms, _schema(), ("jdg:document_job",), docs, {})
    return docs["d1"]["jdg:document_job"]["answer"]


def test_no_refit_keeps_the_frozen_answer():
    assert _run(_Model()) == "this_deal"


def test_refit_add_on_moves_the_answer(tmp_path):
    m = JudgmentRefit(4, 3)
    with torch.no_grad():
        m.net[-1].bias.copy_(torch.tensor([0.0, 0.0, 5.0]))
    path = tmp_path / "jr.pt"
    torch.save({"heads": {"jdg:document_job": {"state": m.state_dict(), "d": 4,
                                               "answers": ["_absent", "this_deal", "other_job"]}}}, path)
    refits = load_refits(str(path))
    assert set(refits) == {"jdg:document_job"}
    assert _run(_Model(refits)) == "other_job"
