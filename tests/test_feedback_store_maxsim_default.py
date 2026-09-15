"""A merged correction scores a query against its nearest exemplar, not the mean."""
import numpy as np

from app.core.decide import DecisionScope
from app.core.feedback_store import Correction, FeedbackStore, SCOPE_GLOBAL

_AXES = {"reset gateway": 0, "update printer ips": 1, "photo documentation": 2, "unrelated": 3}


def _embed(texts):
    out = np.zeros((len(texts), 4), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i, _AXES[t]] = 1.0
    return out


def _store(monkeypatch, env=None):
    if env is None:
        monkeypatch.delenv("SOWSMITH_NEURAL_MAXSIM", raising=False)
    else:
        monkeypatch.setenv("SOWSMITH_NEURAL_MAXSIM", env)
    s = FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True)
    s.add(Correction(
        id="c_task", relation="atom_type", verdict="task", scope=SCOPE_GLOBAL, scope_key="",
        exemplars=["reset gateway", "update printer ips", "photo documentation"], threshold=0.8,
    ))
    return s


def _resolve(s, text):
    return s.resolve(relation="atom_type", text=text, candidates=["task"], context="",
                     scope=DecisionScope(), instruction="", relations=None)


def test_one_of_several_taught_lines_fires_by_default(monkeypatch):
    d = _resolve(_store(monkeypatch), "update printer ips")
    assert d is not None and d.verdict == "task"


def test_unrelated_text_still_abstains(monkeypatch):
    assert _resolve(_store(monkeypatch), "unrelated") is None


def test_mean_prototype_is_one_env_var_away(monkeypatch):
    # Three orthogonal exemplars: each sits at cosine 0.577 from their mean.
    assert _resolve(_store(monkeypatch, env="0"), "update printer ips") is None
