from app.core import semantic_rules as R


def test_collect_records_a_miss_and_answers_lexically(monkeypatch):
    import app.core.embedding_retrieval as E

    monkeypatch.setattr(E, "cached_embedding", lambda _t: None)
    monkeypatch.setattr(R.SemanticRule, "_disabled", staticmethod(lambda: False))
    rule = R.SemanticRule("t_collect", ["a"], ["b"], 0.5, lexical_fallback=lambda t: t == "yes")
    with R.collect_misses() as found:
        assert rule.fires("yes") is True
        assert rule.fires("no") is False
    assert found == ["yes", "no"]
    assert getattr(R._COLLECT, "misses", None) is None
