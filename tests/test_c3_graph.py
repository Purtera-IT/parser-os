"""Run E's cross-line reasoning (ml/c3/graph.py, arch_graph_review.md fixes 1-9)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ml.c3.data import IGNORE, DealExample, featurize  # noqa: E402
from ml.c3.gates import checkpoint_meta, load_checkpoint, save_checkpoint  # noqa: E402
from ml.c3.graph import (GRAPH_DEFAULTS, RUNE_GRAPH, hop, link_weights, parse_graph,  # noqa: E402
                         visibility)
from ml.c3.loop import SMALL, _student_optimizer, gold_weights, rate_lambdas  # noqa: E402
from ml.c3.losses import LossWeights, c3_loss  # noqa: E402
from ml.c3.model import C3Model  # noqa: E402
from ml.c3.schema import GROUP, LINE, PAIR, load_schema  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "ml" / "c3" / "fixtures" / "synthetic_deal.json"
RUNE_SH = Path(__file__).resolve().parents[1] / "ml" / "c3" / "pod" / "runE.sh"
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


def _fixture(schema):
    return featurize(DealExample.load(FIXTURE), schema)


def _open(m, scale=0.5, seed=1):
    """Every zero-init graph map set to a fixed random value (a trained gate)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name in ("link_mix", "link_mix_h", "rel_msg", "node_table", "node_doc", "node_read",
                     "node_back", "pool_slot"):
            mod = getattr(m, name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * scale / max(1, p.shape[-1]) ** 0.5)


# ------------------------------------------------------------- fix 1: the gate group
def test_gate_group_is_in_the_optimizer_from_the_start_and_opens_at_links(schema):
    m = _model(schema, RUNE_GRAPH)
    from ml.c3.loop import _freeze_link_gate

    _freeze_link_gate(m, False)
    opt = _student_optimizer(m, 1e-3, True, None)
    gate = [g for g in opt.param_groups if g.get("gate")]
    assert len(gate) == 1 and any(p is m.link_mix.weight for p in gate[0]["params"])
    assert any(p is m.deal_number.weight for p in gate[0]["params"])
    stages = [("lines", 4), ("links", 4), ("cards", 4), ("numbers", 2), ("deal", 4), ("open_links", 4)]
    main, gate_f = rate_lambdas(stages, parts=2, warm=2, gate_warm_passes=1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [gate_f if g.get("gate") else main for g in opt.param_groups])
    lrs = []
    for s in range(sum(n for _, n in stages)):
        if s == 4:
            _freeze_link_gate(m, True)
        opt.step()          # no grads: a no-op step, as the scheduler expects one first
        sched.step()        # torch >= 2.7: strict zip over the groups; never raises here
        lrs.append((opt.param_groups[0]["lr"], gate[0]["lr"]))
    assert all(g == 0.0 for _, g in lrs[:3])
    # Opened at links (step 4), warmed over one pass (2 steps), then the base rate, constant.
    assert lrs[5][1] == pytest.approx(1e-3) and lrs[-1][1] == pytest.approx(1e-3)
    # The deal stage (steps 14-17) trains at the floor, not the cosine tail.
    assert lrs[15][0] >= 0.3e-3 - 1e-12 and lrs[-1][0] < 0.1e-3


def test_run_base_with_a_scheduler_steps_through_open_links(schema):
    """The tiny loop with AdamW and its schedule walks every student stage (the
    old add_param_group raised on torch >= 2.7 at the first step after it)."""
    from ml.c3.loop import Budget, run_base

    b = _fixture(schema)
    hold = featurize(replace(DealExample.load(FIXTURE), deal_id="synthetic-holdout"), schema)
    budget = Budget(teacher=0, lines=1, links=1, cards=1, numbers=1, deal=1, open_links=1, teach=0)
    model, run = run_base([b], [hold], schema, budget, cfg=CFG, adamw=True, warmup_passes=0.5,
                          graph=RUNE_GRAPH, score_holdout=False)
    assert float(model.link_mix.weight.detach().abs().max()) > 0
    assert "open_links" in run.stage_loss


# ------------------------------------------------------------- fix 2: visibility
def test_visibility_masks():
    t = [1.0, 1.0, 2.0, 3.0]
    idx = visibility(t, 4, "cpu", "index")
    arr = visibility(t, 4, "cpu", "arrival")
    view = visibility(t, 4, "cpu", "view")
    assert idx.tolist() == torch.tril(torch.ones(4, 4, dtype=torch.bool)).tolist()
    # A document that arrives whole sees itself: line 0 reads line 1.
    assert bool(arr[0, 1]) and not bool(arr[0, 2]) and bool(arr[3, 0])
    assert bool(view.all())
    cut = visibility(t, 4, "cpu", "view", cut=1.0)
    # As of an earlier compile: lines before the cut read only up to the cut;
    # a later line reads up to its own arrival.
    assert not bool(cut[0, 2]) and bool(cut[2, 0]) and not bool(cut[2, 3]) and bool(cut[3, 2])


def _deal_with_later_email(text: str) -> dict:
    """A site table at t0 whose rows say nothing about what they are, then an
    email at t1 naming the attachment."""
    atoms = []
    for k in range(4):
        atoms.append({"key": f"r{k}", "text": f"Row {k}: 11{k} Main St Springfield", "entered_at": "2026-04-01T10:00:00Z",
                      "doc_id": "sheet-1", "doc_kind": "file", "section": "Sheet1", "order": k,
                      "table_row": k + 1, "table_col": 1})
    atoms.append({"key": "e0", "text": text, "entered_at": "2026-04-03T10:00:00Z", "doc_id": "email-9",
                  "doc_kind": "email", "section": "body", "order": 0})
    return {"deal_id": "graph-later", "company": "purtera", "atoms": atoms}


def test_a_later_line_changes_an_earlier_reading_only_under_the_view_mask(schema):
    a = featurize(DealExample.from_dict(_deal_with_later_email("The attached list is our 145 stores.")), schema)
    b = featurize(DealExample.from_dict(_deal_with_later_email("The attached sheet is the task list.")), schema)
    for graph, moves in ((GRAPH_DEFAULTS, False), ({**GRAPH_DEFAULTS, "mask": "arrival"}, False),
                         ({**GRAPH_DEFAULTS, "mask": "view"}, True), (RUNE_GRAPH, True)):
        m = _model(schema, graph)
        if graph["hop_both"]:
            _open(m)       # the old hop's causal back-message is the one later->earlier path it had
        with torch.no_grad():
            la = m(a.oneshot()).logits["col:label_type"][:4]
            lb = m(b.oneshot()).logits["col:label_type"][:4]
        assert bool((la - lb).abs().max() > 1e-5) == moves, graph


def test_serve_reads_with_the_mask_training_used(schema):
    """predict.read_deal builds the batch from entered_at exactly as training
    does, so the model computes the same mask: same readings as the training
    forward on the same deal."""
    from ml.c3.predict import read_deal

    deal = DealExample.from_dict(_deal_with_later_email("The attached list is our 145 stores."))
    batch = featurize(deal, schema)
    m = _model(schema, RUNE_GRAPH)
    _open(m)
    with torch.no_grad():
        ref = m(batch.oneshot()).logits["col:label_type"].softmax(-1).max(-1).values
    got = read_deal(m, deal, schema, max_atoms=8192, lines_only=True, heads=("col:label_type",))
    for i, key in enumerate(batch.keys):
        assert got["lines"][key]["reads"]["col:label_type"]["p"] == pytest.approx(float(ref[i]), abs=2e-2)


# ------------------------------------------------------------- fix 3: governs both ways, hop 2
def test_governs_child_reads_parent_parent_reads_children_and_a_second_hop_reaches_siblings():
    n, d = 4, 3
    r = torch.eye(n, d + 1)[:, :d] * 0 + torch.arange(n * d, dtype=torch.float32).view(n, d)
    off = ~torch.eye(n, dtype=torch.bool)
    gov = torch.full((n, n), -20.0)
    gov[0, 1] = gov[0, 2] = gov[0, 3] = 0.0           # log P(child c ⊂ parent 0) = log 1
    w = link_weights({"governs": gov.transpose(0, 1)}, {}, off)["governs"]
    # The child puts its mass on its parent (not the 0.005 the inverted hop gave), the header on nothing.
    assert float(w[1, 0]) > 0.6 and float(w[0].sum()) < 1e-6

    class M:
        rel_msg = None

        @staticmethod
        def link_mix(x):
            return x
    r1 = hop(M, r, {"governs": w}, off)
    # Header reads its rows' mean (normalised by mass + 1), a row reads the header.
    assert not torch.allclose(r1[0], r[0]) and not torch.allclose(r1[1], r[1])
    r_changed = r.clone()
    r_changed[3] += 10.0
    one = hop(M, r_changed, {"governs": w}, off)
    two = hop(M, one, {"governs": w}, off)
    base_two = hop(M, hop(M, r, {"governs": w}, off), {"governs": w}, off)
    assert torch.allclose(one[1], r1[1], atol=1e-5)            # one hop: sibling 3 cannot reach row 1
    assert not torch.allclose(two[1], base_two[1], atol=1e-4)  # two hops: row 3 -> header -> row 1


def test_rows_read_a_header_whose_text_alone_changed(schema):
    """With the gate open, a row's reading follows its header's words (the
    hop and the nodes), under the governs boxes clamped to the header."""
    def deal(head):
        atoms = [{"key": "h", "text": head, "entered_at": "2026-04-01T10:00:00Z", "doc_id": "d", "doc_kind": "file",
                  "section": "S", "order": 0}]
        atoms += [{"key": f"r{k}", "text": f"Item {k}", "entered_at": "2026-04-01T10:00:00Z", "doc_id": "d",
                   "doc_kind": "file", "section": "S", "order": k + 1} for k in range(3)]
        return featurize(DealExample.from_dict({"deal_id": "hdr", "company": "purtera", "atoms": atoms}), schema)
    a, b = deal("Store locations"), deal("Installation tasks")
    m = _model(schema, RUNE_GRAPH)
    _open(m)
    with torch.no_grad():
        ra, rb = m(a.oneshot()).r, m(b.oneshot()).r
    assert float((ra[1:] - rb[1:]).abs().max()) > 1e-4


# ------------------------------------------------------------- fix 4: hop into h
def test_free_heads_see_the_hop(schema):
    b = _fixture(schema)
    m = _model(schema, RUNE_GRAPH)
    _open(m)
    with torch.no_grad():
        m.free_heads["col:label_type"].weight.normal_(0, 0.1)    # a trained free head reads h
        on = m(b.oneshot()).logits["col:label_type"]
        m.link_mix_h.weight.zero_(); m.link_mix_h.bias.zero_()
        off = m(b.oneshot()).logits["col:label_type"]
    assert float((on - off).abs().max()) > 1e-5


# ------------------------------------------------------------- fix 5: pointer edge, citations
def test_pointer_edge_reaches_r_and_citations_train_the_lattice(schema):
    b = _fixture(schema)
    b.hint_lines = [[j for j in range(len(b)) if j != i][:2] for i in range(len(b))]
    m = _model(schema, RUNE_GRAPH).train()
    with torch.no_grad():
        m.rel_msg["rests_on__out"].weight.normal_(0, 0.1)
    loss, parts = c3_loss(m, b, LossWeights(citations=1.0))
    assert parts["citations"] > 0 and parts["pointers"] > 0
    loss.backward()
    assert m.lattice.layers[0].deal_attn.q.weight.grad.abs().sum() > 0
    assert m.cite_q.weight.grad.abs().sum() > 0


# ------------------------------------------------------------- fix 6: near_miss stays out
def test_near_miss_sends_no_message(schema):
    b = _fixture(schema)
    m = _model(schema, RUNE_GRAPH)
    _open(m)
    assert not m.graph["hop_near_miss"]
    with torch.no_grad():
        base = m(b.oneshot()).r
        m.relation_head.__dict__["_orig"] = m.relation_head.forward
        rels = [o.field for o in schema.select(kind="relation") if o.field != "governs"]
        k = rels.index("near_miss")

        def louder(x, d):
            s, z = m.relation_head._orig(x, d)
            s = list(s)
            s[k] = s[k] * 0 + 50.0
            return s, z
        m.relation_head.forward = louder
        moved = m(b.oneshot()).r
    assert torch.allclose(base, moved, atol=1e-5)


# ------------------------------------------------------------- fix 7 + table node: cards and coherence
def _site_table(n_rows=6, kind="physical_site", labeled=3, pointer=True, pointer_text="Attached is the list of our stores."):
    atoms = []
    for k in range(n_rows):
        a = {"key": f"t{k}", "text": f"{100 + k} Oak Road, Unit {k}", "entered_at": "2026-04-01T10:00:00Z",
             "doc_id": "sites-xlsx", "doc_kind": "file", "section": "Stores", "order": k,
             "table_row": k + 1, "table_col": 1}
        if k < labeled:
            a["label"] = {"label_type": kind}
        atoms.append(a)
    edges = []
    if pointer:
        atoms.append({"key": "p", "text": pointer_text, "entered_at": "2026-04-02T10:00:00Z", "doc_id": "mail-1",
                      "doc_kind": "email", "section": "body", "order": 0, "label": {"label_type": "_keep"}})
        edges.append({"src": "p", "dst": "t0", "relation": "governs"})
    return {"deal_id": "sites", "company": "purtera", "atoms": atoms, "edges": edges}


def test_roster_and_sheet_targets_come_from_gold_line_types(schema):
    from ml.c3.graph_losses import node_pointer_targets, table_facts

    b = featurize(DealExample.from_dict(_site_table()), schema)
    m = _model(schema)
    facts = table_facts(m, b)
    roster = [j for j in facts["cards"] if j.key == "jdg:roster"]
    assert len(roster) == 1 and schema.by_key()["jdg:roster"].answers[roster[0].answer].value == "site_roster"
    assert facts["tables"][0]["share"] == 1.0
    tasks = featurize(DealExample.from_dict(_site_table(kind="task")), schema)
    cards = {j.key: schema.by_key()[j.key].answers[j.answer].value for j in table_facts(m, tasks)["cards"]}
    assert cards == {"jdg:roster": "not_site_roster", "jdg:sheet": "scope"}
    # The email naming the attachment points at the table's node (its first row).
    p = b.keys.index("p")
    assert node_pointer_targets(b) == [(p, [b.keys.index("t0")])]


def test_cards_train_queried_lines_and_read_their_slot(schema):
    _, sizes = gold_weights("cards")
    assert set(sizes) == {PAIR, GROUP, LINE}
    assert gold_weights("cards")[0].derived_cards > 0 and gold_weights("links")[0].derived_cards == 0
    b = featurize(DealExample.from_dict(_site_table()), schema)
    m = _model(schema, RUNE_GRAPH).train()
    loss, parts = c3_loss(m, b, judgment_sizes=(PAIR, GROUP, LINE))
    assert parts["derived_cards"] > 0 and parts["table_kind"] > 0 and parts["coherence"] > 0
    assert parts["node_pointers"] > 0
    loss.backward()
    assert m.pool_slot.weight.grad.abs().sum() > 0          # the card reads its space's slot
    assert m.node_q.weight.grad.abs().sum() > 0             # the pointer to the node trains


def _train_tables(schema, steps=900, seed=5):
    """Tiny model, toy deals: rows say nothing about their kind (the same rows
    come as a site list in one deal and a task list in its twin); the email
    naming the attachment, arriving after the table, does. Half the rows are
    labeled; the coherence term reads the table's kind onto the rest."""
    import random

    rnd = random.Random(seed)
    streets = ["Oak", "Pine", "Lake", "Hill", "Mill", "Park", "Bay", "Ash", "Elm", "Cedar"]
    pointers = {"physical_site": ["Attached is the list of our stores.", "The attached sheet is our store list.",
                                  "Here are the store locations in the file."],
                "task": ["Attached is the task list for the install.", "The attached sheet is the work to do.",
                         "Here are the install tasks in the file."]}
    deals = []
    for d in range(10):
        rows = [f"{rnd.randint(10, 999)} {rnd.choice(streets)} Road, Unit {rnd.randint(1, 99)}" for _ in range(6)]
        for kind in ("physical_site", "task"):
            blob = _site_table(n_rows=6, kind=kind, labeled=3, pointer_text=rnd.choice(pointers[kind]))
            blob["deal_id"] = f"toy-{d}-{kind}"
            for a in blob["atoms"]:
                if a["key"].startswith("t"):
                    a["text"] = rows[int(a["key"][1:])]
            deals.append(featurize(DealExample.from_dict(blob), schema))
        # A real mixed table, no pointer: each row's own words decide.
        blob = _site_table(n_rows=6, labeled=0, pointer=False)
        blob["deal_id"] = f"toy-{d}-mixed"
        for k, a in enumerate(blob["atoms"]):
            site = k % 2 == 0
            a["text"] = (f"Store {rnd.randint(1, 999)} {rnd.choice(streets)} Road" if site
                         else f"Install {rnd.choice(['cable', 'rack', 'screen', 'switch'])} at bay {rnd.randint(1, 9)}")
            if k < 4:
                a["label"] = {"label_type": "physical_site" if site else "task"}
        deals.append(featurize(DealExample.from_dict(blob), schema))
    m = _model(schema, RUNE_GRAPH, seed=seed).train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    # Line type weighted as run E weighs it (TYPE_WEIGHT=10), plus the graph's
    # own terms; the rest of the table is off so a few hundred tiny steps are enough.
    from dataclasses import fields

    off = {f.name: 0.0 for f in fields(LossWeights) if f.type in ("float", float)
           and not f.name.endswith(("_p", "_cap", "_margin", "_negative"))}
    w = replace(LossWeights(**off), heads=1.0, coherence=0.5, table_kind=0.5, node_pointers=0.5,
                derived_cards=0.5, head={"col:label_type": 10.0})
    for s in range(steps):
        b = deals[s % len(deals)]
        loss, _ = c3_loss(m, b, w, judgment_sizes=(GROUP,))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


@pytest.fixture(scope="module")
def table_model(schema):
    return _train_tables(schema)


def test_an_ambiguous_site_table_reads_all_sites_when_its_pointer_says_site_list(schema, table_model):
    types = [a.value for a in schema.by_key()["col:label_type"].answers]
    blob = _site_table(n_rows=8, labeled=0, pointer_text="Attached is the list of our stores.")
    for a in blob["atoms"]:
        if a["key"].startswith("t"):
            a["text"] = f"{900 + int(a['key'][1:])} Elm Street, Unit 77"
    b = featurize(DealExample.from_dict(blob), schema)
    with torch.no_grad():
        got = table_model(b.oneshot()).logits["col:label_type"].argmax(-1).tolist()
    rows = [types[got[b.keys.index(f"t{k}")]] for k in range(8)]
    assert rows == ["physical_site"] * 8, rows


def test_a_real_mixed_table_stays_mixed(schema, table_model):
    """Rows whose own words are strong keep their own answers, table node or not."""
    types = [a.value for a in schema.by_key()["col:label_type"].answers]
    blob = _site_table(n_rows=6, labeled=0, pointer=False)
    for k, a in enumerate(blob["atoms"]):
        a["text"] = f"Store {500 + k} Birch Road" if k % 2 == 0 else f"Install rack at bay {k}"
    b = featurize(DealExample.from_dict(blob), schema)
    with torch.no_grad():
        got = table_model(b.oneshot()).logits["col:label_type"].argmax(-1).tolist()
    assert [types[g] for g in got] == ["physical_site", "task"] * 3


# ------------------------------------------------------------- fixes 8, 9: deal, overlay, window, checkpoint
def test_rune_sh_trains_the_deal_stage_validates_at_serves_window_and_turns_the_graph_on():
    text = RUNE_SH.read_text()
    assert "deal=${DEAL:-2}" in text and "GRAPH=${GRAPH:-rune}" in text
    assert "VALID_ATOMS=${VALID_ATOMS:-8192}" in text and "--validation-max-atoms" in text
    assert "question_fit --deals" in text


def test_a_stale_question_overlay_is_refused(tmp_path):
    from ml.c3.question_fit import QuestionRefit, file_md5, stale_refit

    base, other = tmp_path / "a.pt", tmp_path / "b.pt"
    base.write_bytes(b"base")
    other.write_bytes(b"other")
    fit = tmp_path / "refit.pt"
    torch.save({"state": QuestionRefit(4, 3).state_dict(), "d": 4, "answers": ["a", "b", "c"],
                "base_md5": file_md5(str(base))}, fit)
    assert stale_refit(str(fit), str(base)) == ""
    assert "refit it" in stale_refit(str(fit), str(other))
    old = tmp_path / "old.pt"
    torch.save({"state": QuestionRefit(4, 3).state_dict(), "d": 4, "answers": ["a", "b", "c"]}, old)
    assert "no base md5" in stale_refit(str(old), str(base))


def test_checkpoint_carries_the_graph_and_the_training_window(schema, tmp_path):
    from dataclasses import asdict

    from ml.c3.readiness import digest

    m = _model(schema, RUNE_GRAPH)
    _open(m)
    sha = digest([asdict(o) for o in schema.opportunities])
    path = tmp_path / "ck.pt"
    save_checkpoint(path, m, step=1, schema_sha256=sha, max_atoms=2560)
    meta = checkpoint_meta(path)
    assert meta["graph"] == RUNE_GRAPH and meta["max_atoms"] == 2560
    fresh = _model(schema)                    # serve builds the plain model ...
    load_checkpoint(path, fresh, schema_sha256=sha)
    assert fresh.graph == RUNE_GRAPH          # ... and the checkpoint turns the same graph on
    b = _fixture(schema)
    with torch.no_grad():
        assert torch.allclose(fresh(b.oneshot()).logits["col:label_type"],
                              m(b.oneshot()).logits["col:label_type"], atol=1e-5)
    # A default-graph checkpoint stores nothing, so old readers are untouched.
    plain = tmp_path / "plain.pt"
    save_checkpoint(plain, _model(schema), step=1, schema_sha256=sha)
    assert "graph" not in checkpoint_meta(plain)


def test_parse_graph():
    assert parse_graph("rune") == RUNE_GRAPH
    assert parse_graph("rune,mask=arrival")["mask"] == "arrival"
    assert parse_graph("off") == GRAPH_DEFAULTS
    with pytest.raises(ValueError):
        parse_graph("mask=sideways")


def test_serve_returns_table_verdicts(schema):
    from ml.c3.predict import read_deal

    m = _model(schema, RUNE_GRAPH)
    deal = DealExample.from_dict(_site_table())
    got = read_deal(m, deal, schema, max_atoms=8192, lines_only=True, table_heads=("jdg:roster", "jdg:sheet"))
    t = got["tables"]["sites-xlsx||Stores"]
    assert t["lines"] == 6 and {"jdg:roster", "jdg:sheet", "kind"} <= set(t)


def test_masked_rows_keep_gradients_finite(schema):
    """A row with nothing to read (a line in no table, a hint row it cannot
    see) must not turn the backward pass to NaN: the tiny smoke's deal stage
    lost every weight that way."""
    from ml.c3.graph import masked_softmax
    from ml.c3.losses import pointer_loss

    x = torch.randn(3, 4, requires_grad=True)
    mask = torch.tensor([[True, False, True, False], [False] * 4, [True] * 4])
    w = masked_softmax(x, mask)
    assert float(w[1].abs().sum()) == 0.0
    w.sum().backward()
    assert torch.isfinite(x.grad).all()
    blob = _site_table()
    blob["atoms"].append({"key": "loose", "text": "A loose note.", "entered_at": "2026-04-03T10:00:00Z",
                          "label": {"label_type": "_keep"}})
    b = featurize(DealExample.from_dict(blob), schema)
    b.hint_lines = [[len(b) - 1] if i == 0 else [] for i in range(len(b))]
    m = _model(schema, {**RUNE_GRAPH, "mask": "index"}).train()   # line 0 cannot see its hint row
    loss, parts = c3_loss(m, b)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    out = m(b.inputs())
    assert float(pointer_loss(out, b)) == 0.0


def test_no_later_line_leaks_back_under_the_index_mask(schema):
    """Every path (the hop both ways, the nodes and their pointers) keeps the
    mask: under ``index`` the earliest rows' readings ignore a later email."""
    a = featurize(DealExample.from_dict(_deal_with_later_email("The attached list is our 145 stores.")), schema)
    b = featurize(DealExample.from_dict(_deal_with_later_email("The attached sheet is the task list.")), schema)
    m = _model(schema, {**RUNE_GRAPH, "mask": "index"})
    _open(m)
    with torch.no_grad():
        m.free_heads["col:label_type"].weight.normal_(0, 0.1)
        assert torch.allclose(m(a.oneshot()).logits["col:label_type"][:4],
                              m(b.oneshot()).logits["col:label_type"][:4], atol=1e-6)
