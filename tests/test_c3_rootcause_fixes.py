"""Pilot A's three measured causes, fixed behind flags (rootcause.md M1-M2, loss_plan.md "Trunk anchor")."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ml.c3.data import IGNORE, DealExample, featurize
from ml.c3.graph import GRAPH_DEFAULTS, RUNE_GRAPH, parse_graph
from ml.c3.loop import SMALL, _student_optimizer, _student_step
from ml.c3.losses import LossWeights
from ml.c3.model import C3Model
from ml.c3.schema import embedded_doc, load_schema, type_contrast

ROOT = Path(__file__).resolve().parents[1]
RUNE_SH = ROOT / "ml" / "c3" / "pod" / "runE.sh"
CFG = replace(SMALL, asked_layers=1, dropout=0.0, context_dropout=0.0, tag_dropout=0.0)


@pytest.fixture(scope="module")
def schema():
    return load_schema()


def _model(schema, graph=None, seed=0):
    torch.manual_seed(seed)
    m = C3Model(schema, CFG)
    m.add_free_head("col:label_type")
    if graph is not None:
        m.enable_graph(graph)
    return m.eval()


def _two_docs(chatter: str) -> dict:
    """A site sheet, then an email in another document."""
    atoms = [{"key": f"r{k}", "text": f"Store {k}: 11{k} Main St Springfield IL", "entered_at": "2026-04-01T10:00:00Z",
              "doc_id": "sheet-1", "doc_kind": "file", "section": "Sheet1", "order": k} for k in range(3)]
    atoms.append({"key": "e0", "text": chatter, "entered_at": "2026-04-01T09:00:00Z", "doc_id": "email-9",
                  "doc_kind": "email", "section": "body", "order": 0})
    return {"deal_id": "two-docs", "company": "purtera", "atoms": atoms}


# ---------------------------------------------------------------- 1. own-document lattice
def test_lattice_doc_slots_cite_only_their_own_document(schema):
    b = featurize(DealExample.from_dict(_two_docs("Thank you, I'll get the quote over to you.")), schema)
    doc = torch.tensor(b.same_doc, dtype=torch.bool) if not isinstance(b.same_doc, torch.Tensor) else b.same_doc.bool()
    m = _model(schema, {**RUNE_GRAPH, "lattice_doc": True})
    with torch.no_grad():
        out = m(b.oneshot())
    cites = out.citations.sum(1)                                       # [N, N] over spaces
    assert float(cites[~(doc | torch.eye(len(b), dtype=torch.bool))].abs().max()) == 0.0
    off = _model(schema, RUNE_GRAPH)
    with torch.no_grad():
        assert float(off(b.oneshot()).citations.sum(1)[~doc].abs().max()) > 0  # the old lattice reads across


def test_lattice_doc_makes_other_documents_chatter_invisible_to_the_slots(schema):
    """Same h for the site rows, the email's h changed: own-document slots do not move."""
    from types import SimpleNamespace

    b = featurize(DealExample.from_dict(_two_docs("Thank you,")), schema)
    inp = b.oneshot()
    n = len(b)
    rows = [i for i, k in enumerate(b.keys) if k.startswith("r")]
    e = b.keys.index("e0")
    allowed = torch.ones(n, n, dtype=torch.bool)

    def slots(m, h):
        out = SimpleNamespace(h=h)
        with torch.no_grad():
            m._ask(out, inp, m.describe(), None, allowed)
        return out.slots

    torch.manual_seed(7)
    h = torch.randn(n, CFG.d)
    h2 = h.clone()
    h2[e] = torch.randn(CFG.d) * 3
    on = _model(schema, {**GRAPH_DEFAULTS, "mask": "view", "lattice_doc": True})
    off = _model(schema, {**GRAPH_DEFAULTS, "mask": "view"})
    off.load_state_dict(on.state_dict())
    assert torch.equal(slots(on, h)[rows], slots(on, h2)[rows])
    assert not torch.allclose(slots(off, h)[rows], slots(off, h2)[rows])


def test_lattice_doc_travels_in_the_checkpoint(schema, tmp_path):
    from dataclasses import asdict

    from ml.c3.gates import checkpoint_meta, load_checkpoint, save_checkpoint
    from ml.c3.readiness import digest

    m = _model(schema, parse_graph("rune,lattice_doc=1"))
    sha = digest([asdict(o) for o in schema.opportunities])
    save_checkpoint(tmp_path / "ck.pt", m, step=1, schema_sha256=sha)
    assert checkpoint_meta(tmp_path / "ck.pt")["graph"]["lattice_doc"] is True
    fresh = _model(schema)
    load_checkpoint(tmp_path / "ck.pt", fresh, schema_sha256=sha)
    assert fresh.graph["lattice_doc"] is True                     # serve reads with the same mask
    b = featurize(DealExample.from_dict(_two_docs("Thank you,")), schema)
    with torch.no_grad():
        assert torch.allclose(fresh(b.oneshot()).logits["col:label_type"], m(b.oneshot()).logits["col:label_type"],
                              atol=1e-5)
    assert parse_graph("rune")["lattice_doc"] is False and GRAPH_DEFAULTS["lattice_doc"] is False


def test_citation_loss_ignores_hints_the_slot_cannot_reach(schema):
    from ml.c3.graph_losses import citation_loss

    b = featurize(DealExample.from_dict(_two_docs("Thank you,")), schema)
    e = b.keys.index("e0")
    b.hint_lines = [[e] if k.startswith("r") else [] for k in b.keys]      # every hint in the other document
    m = _model(schema, {**RUNE_GRAPH, "lattice_doc": True})
    out = m(b.oneshot())
    assert float(citation_loss(m, out, b)) == 0.0


def test_runE_turns_the_own_document_lattice_on():
    sh = RUNE_SH.read_text()
    assert 'LATTICE_DOC=${LATTICE_DOC:-1}' in sh and 'GRAPH="$GRAPH,lattice_doc=1"' in sh


# ---------------------------------------------------------------- 2. trunk anchor
def _labeled_fixture(schema):
    """The synthetic deal with every other line's type unlabeled (what the KL anchor holds)."""
    deal = DealExample.load(ROOT / "ml" / "c3" / "fixtures" / "synthetic_deal.json")
    b = featurize(deal, schema)
    t = b.targets["col:label_type"]
    b.targets = {**b.targets, "col:label_type": [IGNORE if i % 2 else y for i, y in enumerate(t)]}
    return b


def test_trunk_group_has_its_own_rate_and_no_decay_toward_zero(schema):
    from ml.c3.anchor import trunk_ids

    m = _model(schema).train()
    opt = _student_optimizer(m, 1e-3, True, None, trunk_lr_scale=0.1, trunk_decay=0.0)
    trunk = [g for g in opt.param_groups if g.get("trunk")]
    assert len(trunk) == 1 and trunk[0]["lr"] == pytest.approx(1e-4) and trunk[0]["weight_decay"] == 0.0
    tid = trunk_ids(m)
    assert tid and {id(p) for p in trunk[0]["params"]} == tid
    assert not any(id(p) in tid for g in opt.param_groups if not g.get("trunk") for p in g["params"])
    # Off: one group as before.
    assert not any(g.get("trunk") for g in _student_optimizer(m, 1e-3, True, None).param_groups)


def test_pull_holds_the_trunk_near_init_and_kl_holds_unlabeled_predictions(schema):
    from ml.c3.anchor import TrunkAnchor

    b = _labeled_fixture(schema)
    w = replace(LossWeights(), head={"col:label_type": 5.0})

    def run(pull, kl, scale):
        m = _model(schema, seed=3).train()
        ruler = TrunkAnchor(m, pull=1.0)                     # only measures the drift
        anchor = TrunkAnchor(m, pull=pull, kl=kl, heads=("col:label_type",))
        anchor.cache(m, [b])
        with torch.no_grad():
            m.eval()
            init_lp = m(b.inputs()).logits["col:label_type"].log_softmax(-1)
            m.train()
        opt = _student_optimizer(m, 3e-3, True, None, trunk_lr_scale=scale, trunk_decay=0.0 if pull else None)
        for _ in range(12):
            _student_step(m, opt, b, w, None, None, None, "cpu", anchor=anchor if (pull or kl) else None)
        m.eval()
        with torch.no_grad():
            lp = m(b.inputs()).logits["col:label_type"].log_softmax(-1)
        free = [i for i, y in enumerate(b.targets["col:label_type"]) if y == IGNORE]
        assert free
        return ruler.drift(m), float(torch.nn.functional.kl_div(lp[free], init_lp[free], log_target=True,
                                                                reduction="batchmean"))

    loose_drift, loose_kl = run(0.0, 0.0, 1.0)
    held_drift, held_kl = run(0.2, 2.0, 0.1)
    assert held_drift < 0.5 * loose_drift
    assert held_kl <= loose_kl + 1e-6


def test_anchor_survives_a_resume(schema, tmp_path):
    from ml.c3.anchor import TrunkAnchor

    b = _labeled_fixture(schema)
    m = _model(schema)
    a = TrunkAnchor(m, pull=0.01, kl=0.5, heads=("col:label_type",))
    assert a.cache(m, [b]) > 0
    a.save(tmp_path / "a.pt")
    with torch.no_grad():
        for p in m.encoder.blocks.parameters():
            p.add_(1.0)
    b2 = TrunkAnchor(m, pull=0.01, kl=0.5, heads=("col:label_type",))     # snapshot of the moved weights...
    assert b2.load(tmp_path / "a.pt")                                     # ...replaced by INIT's
    assert b2.drift(m) > 0.01 and set(b2.targets) == set(a.targets)


def test_loop_refuses_a_new_head_as_an_anchor():
    r = subprocess.run([sys.executable, "-m", "ml.c3.loop", "--tiny", "--train", "x.json",
                        "--anchor-heads", "col:label_type,read:visit_count", "--anchor-kl", "0.5",
                        "--allow-new-heads", "read:visit_count"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "INIT has no prediction for a new head" in r.stderr


def test_runE_sets_the_trunk_anchor():
    sh = RUNE_SH.read_text()
    for k in ("TRUNK_LR_SCALE=${TRUNK_LR_SCALE:-0.1}", "TRUNK_PULL=${TRUNK_PULL:-0.01}",
              "ANCHOR_KL=${ANCHOR_KL:-0.5}", "--trunk-lr-scale", "--trunk-pull", "--anchor-kl", "--anchor-heads"):
        assert k in sh, k


# ---------------------------------------------------------------- 3. descriptions
def _unpinned(monkeypatch):
    monkeypatch.setenv("C3_TEXT_PINS", "0")
    return load_schema()


def test_embedded_type_texts_carry_no_contrast_clauses(schema, monkeypatch):
    types = {a.value: a.description for a in _unpinned(monkeypatch).by_key()["col:label_type"].answers}
    for v, t in types.items():
        assert "unlike" not in t.lower(), v
    assert "site attribute" not in types["physical_site"] and "stakeholder" not in types["physical_site"]
    assert types["physical_site"].startswith("physical site: A place the work happens")
    assert "is a payment term" not in types["contract_term"] and "would be a task" not in types["dependency"]
    assert types["_keep"].startswith("noise: ") and "this is the rest" not in types["_keep"].lower()


def test_the_contrast_stays_in_the_registry_as_guidance():
    c = type_contrast()
    assert any(x.startswith("Unlike a site attribute") for x in c["physical_site"])
    assert "Payment timing is a payment term." in c["contract_term"]
    raw = json.loads((ROOT / "app" / "core" / "atom_types.json").read_text())
    site = next(t for t in raw["types"] if t["name"] == "physical_site")
    assert "Unlike a site attribute" in site["desc"]                     # labelers still read it


def test_embedded_doc_keeps_the_definition():
    assert embedded_doc("A thing. Unlike a widget, it is round; unlike a gadget, it rolls.", ["widget"]) == "A thing."
    assert embedded_doc("A term. Invoice timing is a payment term; cancellation is a change-order rule.",
                        ["payment term", "change order rule"]) == "A term."
    assert embedded_doc("A list of stores, not a quantity.", ["quantity"]) == "A list of stores, not a quantity."


def test_c3_embed_contrast_restores_the_b395fbb_texts():
    code = ("from ml.c3.schema import load_schema; t={a.value:a.description for a in "
            "load_schema().by_key()['col:label_type'].answers}; print('Unlike a site attribute' in t['physical_site'])")
    on = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                        env={**os.environ, "C3_EMBED_CONTRAST": "1", "C3_TEXT_PINS": "0"})
    assert on.stdout.strip() == "True", on.stderr


def test_gate_pins_read_init_texts_and_only_those(schema, monkeypatch):
    """runE/text_pins.json: the heads the description gate failed read INIT's words."""
    from ml.c3.schema import INIT_TEXTS_PATH, TEXT_PINS_PATH

    pins = json.loads(TEXT_PINS_PATH.read_text())
    init = {k: (q, ans) for k, q, ans in json.loads(INIT_TEXTS_PATH.read_text())["texts"]}
    by_key = schema.by_key()
    assert "jdg:site_role" in pins["heads"] and pins["answers"] == {"col:label_type": ["physical_site"]}
    for k in pins["heads"]:
        q, ans = init[k]
        assert by_key[k].description == q and [a.description for a in by_key[k].answers] == [t for _, t in ans], k
    lt = {a.value: a.description for a in by_key["col:label_type"].answers}
    assert lt["physical_site"] == dict(init["col:label_type"][1])["physical_site"]
    assert "Unlike" not in lt["task"]                                   # the rest of the type texts stay new
    free = _unpinned(monkeypatch).by_key()
    assert free["jdg:site_role"].description != init["jdg:site_role"][0]   # C3_TEXT_PINS=0: every new text


def test_description_gate_reads_the_same_weights_under_both_texts(schema):
    from ml.c3.desc_gate import compare, failures, old_description, schema_texts

    m = _model(schema)
    b = _labeled_fixture(schema)
    same = schema_texts(schema)
    rep = compare(m, schema, same, [b], {})
    assert rep["changed"] == [] and rep["heads"]
    assert all(r["old"] == r["new"] for r in rep["heads"].values())
    # A rewrite of the type answers moves the prototypes: the gate sees a different reading.
    moved = [[k, q, [[v, t + " Unlike every other kind of line here."] for v, t in ans] if k == "col:label_type" else ans]
             for k, q, ans in same]
    desc, keys = old_description(m, schema, moved)
    assert "col:label_type" in keys
    assert not torch.allclose(desc["col:label_type"][1], m.describe()["col:label_type"][1])


def test_description_gate_fails_on_a_drop_over_two_points():
    from ml.c3.desc_gate import failures

    rep = {"heads": {"col:label_type": {"old": 0.80, "new": 0.785, "n": 400},
                     "col:supplier": {"old": 0.70, "new": 0.60, "n": 10}},
           "sites": {"010358": {"old": 89, "new": 37, "rows": 101}, "010447": {"old": 237, "new": 236, "rows": 283}}}
    bad = failures(rep, 0.02, 20)
    assert bad == ["sites 010358: 89 -> 37 of 101"]                       # 89 -> 37 blocks; n=10 is not gated
    rep["heads"]["col:label_type"]["new"] = 0.77
    assert any(x.startswith("col:label_type") for x in failures(rep, 0.02, 20))


def test_runE_runs_the_description_gate_before_the_loop():
    sh = RUNE_SH.read_text()
    assert "ml.c3.desc_gate" in sh and "C3_EMBED_CONTRAST" in sh
    assert sh.index("ml.c3.desc_gate") < sh.index("python -m ml.c3.loop")
