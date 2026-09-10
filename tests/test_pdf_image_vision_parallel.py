"""PDF image-vision parallelism: concurrency is DRESS, output is invariant.

The stage was strictly sequential — one VLM call at a time against a shared
Ollama fleet, up to 90s each. It now runs the per-image work in a small bounded
pool. That is only legitimate if the pool is invisible in the result, so this
file pins the invariant the same way ``test_graph_packet_invariance.py`` does:
build a fixture, assert it is non-vacuous, then assert the signature does not
move when the dress (here: ``SOWSMITH_PDF_IMAGE_CONCURRENCY``) changes.

The VLM is monkeypatched with DELIBERATE JITTER — later images return first —
so completion order provably differs from work-list order. Without that the
tests would pass on a pool that accidentally ran in order and prove nothing.
"""
import threading
import time
import types

import pytest

from app.core import pdf_image_vision as piv

CONCURRENCIES = (1, 2, 4, 8)


# ── fixture ─────────────────────────────────────────────────────────


def _marker(tmp_path, i, *, body=None):
    """One image marker with a crop unique to ``i`` (so nothing dedups)."""
    name = f"page{i}_image1.png"
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n" + (body if body is not None else f"crop-{i}|".encode()) * 900)
    return types.SimpleNamespace(
        id=f"atm_marker_{i}",
        project_id="proj1",
        artifact_id="art1",
        parser_version="vtest",
        value={"kind": "image_marker", "region_ref": f"page{i}/image1",
               "saved_path": str(p), "expected_content": f"Figure {i}: rack and battery charger"},
        source_refs=[types.SimpleNamespace(filename="install_guide.pdf")],
    )


def _markers(tmp_path, n):
    return [_marker(tmp_path, i) for i in range(n)]


def _jittered_vlm(*, order_reversing=True, record=None):
    """A VLM mock that makes completion order fight work-list order.

    Image *i* sleeps ``(n-i)`` ticks, so image 0 — first in the work list — is
    the LAST to come back. Any code that assembled output in completion order
    would visibly reverse.
    """
    def _impl(image_bytes, prompt, *, model=None, max_tokens=0):
        idx = int(image_bytes.split(b"crop-", 1)[1].split(b"|", 1)[0])
        if record is not None:
            record.append(idx)
        if order_reversing:
            time.sleep((20 - idx) * 0.002)
        if "triaging" in prompt:
            return '{"image_kind": "photo", "has_text": false, "meaningful": true}'
        if "describing an image" in prompt:
            return ('{"description": "Battery charger %d mounted on the rack",'
                    ' "facts": [{"kind": "equipment", "text": "charger unit %d"}]}' % (idx, idx))
        return "{}"
    return _impl


def _base(monkeypatch):
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setattr(piv, "_vision_reachable", lambda: True)
    monkeypatch.setattr(piv, "_page_context", lambda *a, **k: ("", "", "", 0))
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "")


def _sig(atoms):
    """Everything a downstream consumer can see. Order is part of it."""
    return [(
        a.id, a.raw_text, a.normalized_text, str(a.atom_type),
        a.confidence, tuple(sorted(a.review_flags)),
        tuple(sorted((k, str(v)) for k, v in a.value.items())),
        tuple((s.id, s.filename, str(sorted(s.locator.items()))) for s in a.source_refs),
    ) for a in atoms]


def _run_at(monkeypatch, tmp_path_factory, width, n=8, vlm=None):
    """One full compile at a given concurrency, on its own fresh crops."""
    tmp = tmp_path_factory.mktemp(f"c{width}")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_CONCURRENCY", str(width))
    return piv.process_image_markers(_markers(tmp, n))


# ── THE test: concurrency is invisible ──────────────────────────────


def test_output_is_byte_identical_across_concurrency(monkeypatch, tmp_path_factory):
    """Same input, any pool width, same atoms — ids, order and values."""
    _base(monkeypatch)
    monkeypatch.setattr(piv, "_vlm", _jittered_vlm())

    base = _sig(_run_at(monkeypatch, tmp_path_factory, 1))
    # Non-vacuity: the fixture must actually produce atoms from every image,
    # or "identical" would just be comparing two empty lists.
    assert len(base) == 16, base          # 8 images x (description + 1 fact)
    assert len({s[0] for s in base}) == 16  # 16 distinct atom ids

    for width in CONCURRENCIES[1:]:
        assert _sig(_run_at(monkeypatch, tmp_path_factory, width)) == base, width


def test_atoms_are_in_work_list_order_not_completion_order(monkeypatch, tmp_path_factory):
    """Directly pins the reassembly rule: the mock returns image 7 first and
    image 0 last, and the output still reads 0..7."""
    _base(monkeypatch)
    finished = []
    monkeypatch.setattr(piv, "_vlm", _jittered_vlm(record=finished))

    out = _run_at(monkeypatch, tmp_path_factory, 4)
    descriptions = [a.raw_text for a in out if a.value["fact_kind"] == "image_description"]
    assert descriptions == [f"Battery charger {i} mounted on the rack" for i in range(8)]
    # Non-vacuity for the jitter itself: the pool really did overlap and really
    # did finish out of order, so the assertion above had something to catch.
    assert finished != sorted(finished), finished


def test_multiple_threads_are_actually_used(monkeypatch, tmp_path_factory):
    """Guards against a silent regression to sequential: the whole speedup
    claim rests on more than one image being in flight at a time."""
    _base(monkeypatch)
    seen_threads = set()
    live = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def _vlm(image_bytes, prompt, *, model=None, max_tokens=0):
        with lock:
            seen_threads.add(threading.current_thread().name)
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.02)
        with lock:
            live["now"] -= 1
        if "triaging" in prompt:
            return '{"image_kind": "photo", "has_text": false, "meaningful": true}'
        return '{"description": "a rack photo on the wall", "facts": []}'

    monkeypatch.setattr(piv, "_vlm", _vlm)
    _run_at(monkeypatch, tmp_path_factory, 4)
    assert len(seen_threads) > 1, seen_threads
    assert live["peak"] > 1, live


def test_sequential_width_uses_no_pool(monkeypatch, tmp_path_factory):
    """Concurrency 1 must stay literally sequential — it is the control arm of
    the invariance test, so it may not quietly become a 1-worker pool."""
    _base(monkeypatch)
    monkeypatch.setattr(piv, "_vlm", _jittered_vlm(order_reversing=False))
    main = threading.current_thread().name
    threads = set()
    real = piv._process_one
    monkeypatch.setattr(piv, "_process_one", lambda **kw: (
        threads.add(threading.current_thread().name) or real(**kw)))
    out = _run_at(monkeypatch, tmp_path_factory, 1, n=4)
    assert out                       # non-vacuous: work actually ran
    assert threads == {main}, threads


# ── selection phase: cheap, serial, and unchanged ───────────────────


def test_work_list_phase_makes_no_vlm_calls(monkeypatch, tmp_path):
    """Dedup / cap / min-bytes are decided before a single token is spent."""
    _base(monkeypatch)
    calls = []
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k: calls.append(1) or "{}")
    work = piv._build_work_list(_markers(tmp_path, 6), max_images=40, min_bytes=3000)
    assert len(work) == 6            # non-vacuous: it really selected images
    assert calls == []               # ...and the mock was never touched


def test_dedup_drops_identical_crops_at_every_concurrency(monkeypatch, tmp_path):
    """Identical bytes under different region_refs are ONE image, whatever the
    pool width — the dedup set lives in the serial phase."""
    _base(monkeypatch)
    monkeypatch.setattr(piv, "_vlm", _jittered_vlm())
    same = b"crop-3|"
    for width in CONCURRENCIES:
        monkeypatch.setenv("SOWSMITH_PDF_IMAGE_CONCURRENCY", str(width))
        markers = [_marker(tmp_path, i, body=same) for i in range(5)]
        work = piv._build_work_list(markers, max_images=40, min_bytes=3000)
        assert [w.region_ref for w in work] == ["page0/image1"], width
        out = piv.process_image_markers(markers)
        assert len(out) == 2, width  # one description + one fact, once


def test_cap_picks_the_same_images_as_before(monkeypatch, tmp_path):
    """The ``max_images`` cap still takes a PREFIX of marker order — not
    whichever images a pool happened to grab first."""
    _base(monkeypatch)
    monkeypatch.setattr(piv, "_vlm", _jittered_vlm())
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_MAX", "3")
    markers = _markers(tmp_path, 9)
    work = piv._build_work_list(markers, max_images=3, min_bytes=3000)
    assert [w.region_ref for w in work] == [f"page{i}/image1" for i in range(3)]

    for width in CONCURRENCIES:
        monkeypatch.setenv("SOWSMITH_PDF_IMAGE_CONCURRENCY", str(width))
        out = piv.process_image_markers(_markers(tmp_path, 9))
        got = [a.raw_text for a in out if a.value["fact_kind"] == "image_description"]
        assert got == [f"Battery charger {i} mounted on the rack" for i in range(3)], width


def test_min_bytes_filter_still_applies_in_the_work_list(monkeypatch, tmp_path):
    _base(monkeypatch)
    big = _marker(tmp_path, 0)
    tiny = _marker(tmp_path, 1)
    (tmp_path / "page1_image1.png").write_bytes(b"\x89PNG")
    work = piv._build_work_list([tiny, big], max_images=40, min_bytes=3000)
    assert [w.region_ref for w in work] == ["page0/image1"]


# ── failure containment ─────────────────────────────────────────────


def test_one_failing_image_loses_only_itself(monkeypatch, tmp_path_factory):
    """A blowup on image 3 must not fail the compile, must not drop images
    0-2 or 4-7, and must not shift anyone's position in the output."""
    _base(monkeypatch)
    clean = _jittered_vlm()

    def _vlm(image_bytes, prompt, *, model=None, max_tokens=0):
        idx = int(image_bytes.split(b"crop-", 1)[1].split(b"|", 1)[0])
        if idx == 3:
            raise RuntimeError("VLM host exploded on image 3")
        return clean(image_bytes, prompt, model=model, max_tokens=max_tokens)

    monkeypatch.setattr(piv, "_vlm", _vlm)
    for width in CONCURRENCIES:
        out = _run_at(monkeypatch, tmp_path_factory, width)
        got = [a.raw_text for a in out if a.value["fact_kind"] == "image_description"]
        assert got == [f"Battery charger {i} mounted on the rack"
                       for i in (0, 1, 2, 4, 5, 6, 7)], width


def test_every_image_failing_still_returns_cleanly(monkeypatch, tmp_path_factory):
    _base(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("host down")

    monkeypatch.setattr(piv, "_vlm", _boom)
    for width in CONCURRENCIES:
        assert _run_at(monkeypatch, tmp_path_factory, width) == []


# ── silver logging: exactly once, no loss, no duplication ───────────


def _fresh_log():
    from app.core.training_log import TrainingLog, set_training_log
    log = TrainingLog(":memory:")
    set_training_log(log)
    return log


def _clear_log():
    from app.core.training_log import set_training_log
    set_training_log(None)


def _skip_vlm(image_bytes, prompt, *, model=None, max_tokens=0):
    """Everything triages as a logo — one silver row per image, no describe."""
    idx = int(image_bytes.split(b"crop-", 1)[1].split(b"|", 1)[0])
    time.sleep((20 - idx) * 0.002)
    return '{"image_kind": "logo", "has_text": false, "meaningful": false}'


@pytest.mark.parametrize("width", CONCURRENCIES)
def test_silver_rows_written_exactly_once_under_concurrency(
    monkeypatch, tmp_path_factory, width,
):
    """Every image's gate row lands, once. Concurrent SQLite writes would either
    lose rows (this module swallows logging errors) or double them."""
    _base(monkeypatch)
    # Distinct captions -> distinct content-hash ids, so a LOST row and a
    # DEDUPED row are distinguishable; a shared caption could hide loss.
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "18 total data outlets")
    monkeypatch.setattr(piv, "_vlm", _skip_vlm)
    log = _fresh_log()
    try:
        tmp = tmp_path_factory.mktemp(f"silver{width}")
        monkeypatch.setenv("SOWSMITH_PDF_IMAGE_CONCURRENCY", str(width))
        assert piv.process_image_markers(_markers(tmp, 8)) == []
        rows = log.rows(relation="pdf_image_kind")
        assert len(rows) == 8, [r.id for r in rows]
        assert len({r.id for r in rows}) == 8
        assert {r.provenance["region_ref"] for r in rows} == {
            f"page{i}/image1" for i in range(8)}
        assert all(r.label == "skip" and r.teacher == "llm" for r in rows)
    finally:
        _clear_log()


def test_silver_rows_are_written_in_work_list_order(monkeypatch, tmp_path_factory):
    """The replay is serial AND ordered, so a recompile's log reads the same
    both times regardless of which VLM call returned first."""
    _base(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "18 total data outlets")
    monkeypatch.setattr(piv, "_vlm", _skip_vlm)
    log = _fresh_log()
    try:
        tmp = tmp_path_factory.mktemp("silverorder")
        monkeypatch.setenv("SOWSMITH_PDF_IMAGE_CONCURRENCY", "4")
        piv.process_image_markers(_markers(tmp, 8))
        regions = [r.provenance["region_ref"] for r in log.rows(relation="pdf_image_kind")]
        assert regions == [f"page{i}/image1" for i in range(8)]
    finally:
        _clear_log()


def test_recompile_still_upserts_under_concurrency(monkeypatch, tmp_path):
    """Content-hash ids must keep collapsing recompiles into one row — the pool
    must not turn a re-run into a second copy."""
    _base(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "18 total data outlets")
    monkeypatch.setattr(piv, "_vlm", _skip_vlm)
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_CONCURRENCY", "4")
    log = _fresh_log()
    try:
        markers = _markers(tmp_path, 5)
        assert piv.process_image_markers(markers) == []
        assert piv.process_image_markers(_markers(tmp_path, 5)) == []
        assert log.count(relation="pdf_image_kind") == 5
    finally:
        _clear_log()


# ── thumbnail budget: a global counter that must stay deterministic ──


def _real_marker(tmp_path, i):
    """A marker whose crop is a REAL image, so a thumbnail can be made."""
    import io as _io
    import random as _random
    from PIL import Image
    m = _marker(tmp_path, i)
    rnd = _random.Random(5)
    img = Image.new("RGB", (600, 400))
    img.putdata([((x * 7 + y * 3) % 256, (x * 3) % 256, rnd.randrange(256))
                 for y in range(400) for x in range(600)])
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    # Keep the index marker readable by the jitter mock, and keep bytes unique.
    (tmp_path / f"page{i}_image1.png").write_bytes(
        buf.getvalue() + b"crop-%d|" % i)
    return m


def _veto_setup(monkeypatch):
    from app.core import pdf_image_veto
    monkeypatch.setattr(pdf_image_veto, "enabled", lambda: True)
    monkeypatch.setattr(pdf_image_veto, "veto", lambda c, o: 0.93)


@pytest.mark.parametrize("width", CONCURRENCIES)
def test_thumb_budget_goes_to_the_same_images_at_any_concurrency(
    monkeypatch, tmp_path_factory, width,
):
    """``_thumb_budget`` is a per-compile read-then-increment counter — the one
    piece of global state a pool could both corrupt AND reorder. The budget must
    land on the FIRST images in work-list order, not the fastest ones."""
    _base(monkeypatch)
    _veto_setup(monkeypatch)
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: "18 total data outlets")
    monkeypatch.setattr(piv, "_vlm", _skip_vlm)
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_THUMB_MAX", "2")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_CONCURRENCY", str(width))
    log = _fresh_log()
    try:
        tmp = tmp_path_factory.mktemp(f"thumb{width}")
        markers = [_real_marker(tmp, i) for i in range(6)]
        assert piv.process_image_markers(markers) == []
        verdicts = [m.value["gate_verdict"] for m in markers]
        assert [("crop_thumb" in gv) for gv in verdicts] == [
            True, True, False, False, False, False], width
        for gv in verdicts[2:]:
            assert gv["crop_thumb_error"] == "budget_exhausted", width
        # Non-vacuity: every image was disputed, so all six competed for budget.
        assert all(gv["veto"]["meaningful_prob"] == 0.93 for gv in verdicts)
        assert log.count(relation="pdf_image_veto") == 6
        assert piv._thumb_budget["used"] == 2
    finally:
        _clear_log()


# ── config ──────────────────────────────────────────────────────────


def test_concurrency_default_and_clamps(monkeypatch):
    monkeypatch.delenv("SOWSMITH_PDF_IMAGE_CONCURRENCY", raising=False)
    assert piv._concurrency() == 4
    for raw, want in [("1", 1), ("8", 8), ("0", 1), ("-3", 1), ("999", 16),
                      ("nonsense", 4)]:
        monkeypatch.setenv("SOWSMITH_PDF_IMAGE_CONCURRENCY", raw)
        assert piv._concurrency() == want, raw
