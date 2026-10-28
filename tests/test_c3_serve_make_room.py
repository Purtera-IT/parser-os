import json
from unittest import mock

from ml.c3 import serve


def test_small_reads_leave_the_embedding_model_loaded(monkeypatch):
    monkeypatch.setenv("C3_EMBED_MODEL", "qwen3-embedding:8b")
    with mock.patch("urllib.request.urlopen") as op:
        assert serve._make_room(1200) is False
        op.assert_not_called()


def test_a_big_read_unloads_the_embedding_model_first(monkeypatch):
    monkeypatch.setenv("C3_EMBED_MODEL", "qwen3-embedding:8b")
    monkeypatch.setenv("C3_UNLOAD_EMBED_LINES", "4000")
    ps = mock.Mock(read=lambda: json.dumps({"models": [{"name": "qwen3-embedding:8b"}]}).encode())
    with mock.patch("urllib.request.urlopen", side_effect=[ps, mock.Mock(read=lambda: b"{}")]) as op:
        assert serve._make_room(6364) is True
        req = op.call_args_list[1][0][0]
        assert req.full_url.endswith("/api/embed")
        assert json.loads(req.data)["keep_alive"] == 0


def test_a_big_read_loads_nothing_when_the_model_is_not_loaded(monkeypatch):
    monkeypatch.setenv("C3_EMBED_MODEL", "qwen3-embedding:8b")
    ps = mock.Mock(read=lambda: b'{"models": []}')
    with mock.patch("urllib.request.urlopen", side_effect=[ps]) as op:
        assert serve._make_room(6364) is False
        assert op.call_count == 1


def test_zero_turns_it_off_and_a_failed_unload_never_blocks_the_read(monkeypatch):
    monkeypatch.setenv("C3_EMBED_MODEL", "qwen3-embedding:8b")
    monkeypatch.setenv("C3_UNLOAD_EMBED_LINES", "0")
    assert serve._make_room(9000) is False
    monkeypatch.setenv("C3_UNLOAD_EMBED_LINES", "4000")
    with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
        assert serve._make_room(9000) is False
