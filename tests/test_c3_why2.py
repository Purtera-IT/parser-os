"""Run E why2 (arch_review §5-8), on invented rows: free rows on r, the text paths standing
alone, the unused notes training their heads, the company layer apart from the base.

* an r-only free row starts from the old row's columns (reads exactly as before at
  h_keep 1), reads no h at h_keep 0, and survives a checkpoint round trip;
* ``free_h_out`` sets h_keep each step; ``company_r_grad`` scales the company
  gradient into the trunk (the company layer itself learns at the full rate);
* company free rows read the conduct line only;
* described_alone scores the described logits without the free row;
* card notes (a line card, a reason-only card, a field note) train their own head
  (card_alone, card_contrast); [purtera] lines train only the company layer
  (co_why_contrast / co_why_alone); numbers and pair cards get WHY-alone terms;
* why_swap reads no h;
* the pretend-new-company refit leaves the base bit-identical;
* the admission-purity check, the WHY bank (off by default, prior-balanced, gated)
  and the head-question draft.
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ml.c3 import free_rows, why2, why_bank  # noqa: E402
from ml.c3.company_refit import company_param_names, pretend_new_company  # noqa: E402
from ml.c3.data import DealExample, featurize  # noqa: E402
from ml.c3.gates import load_checkpoint, save_checkpoint  # noqa: E402
from ml.c3.losses import LossWeights, c3_loss  # noqa: E402
from ml.c3.model import C3Config, C3Model  # noqa: E402
from ml.c3.purity import admission_impurities  # noqa: E402
from ml.c3.readiness import digest  # noqa: E402
from ml.c3.schema import apply_guidance, load_schema  # noqa: E402

SMALL = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                 residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
ASKED = replace(SMALL, asked_layers=1)
PM = "pm@example.com"
TYPE = "col:label_type"
REPO = Path(__file__).resolve().parents[1]

ATOMS = [
    {"label_key": f"k{n}", "atom_id": f"at-k{n}", "text": t, "doc_id": "email-1", "doc_kind": "email",
     "section": "body", "entered_at": f"2026-05-01T10:0{n}:00Z", "speaker_role": "customer",
     "speaker_side": "customer"}
    for n, t in enumerate(["Store 12 | 100 Main St | Springfield",
                           "Install two displays at each store.",
                           "Send the invoice to our head office on Main Avenue.",
                           "Store 14 | 9 Elm Rd | Shelbyville",
                           "Thanks, talk soon.",
                           "Saturday installs only."])
]
LABELS = [
    {"label_key": "k0", "label_type": "physical_site", "note": "A street address where crews go.\n"
     "[purtera] We quote travel from the nearest tech.", "reads_set": {"co_action": "keep"}},
    {"label_key": "k1", "label_type": "task", "note": "Crew work at every store.",
     "reads_set": {"labor_hours": 4}},
    {"label_key": "k2", "label_type": "physical_site", "note": "An address where nobody works.",
     "reads_set": {"co_action": "ignore"}},
    {"label_key": "k3", "label_type": "physical_site", "note": "Another address crews go to.\n"
     "[purtera] Second store is in our core region.", "reads_set": {"co_action": "keep"}},
    {"label_key": "k4", "label_type": "small_talk", "rejected": "true", "note": "A sign-off.",
     "reads_set": {"co_action": "reject"}},
    {"label_key": "k5", "label_type": "constraint", "note": "Only one day of the week is open.",
     "reads_set": {"labor_hours": 2}},
]
JUDGMENTS = [
    {"head": "site_role", "verdict": "job_site", "labeler": PM,
     "target": {"site": {"evidence": {"atomId": "at-k0"}}}, "note": "Technicians install at this store."},
    {"head": "site_role", "verdict": "job_site", "labeler": PM,
     "target": {"site": {"evidence": {"atomId": "at-k3"}}}, "note": "Work is planned at the second store."},
    # A reason-only card: its code is the only class target, its note was dropped before why2.
    {"head": "site_role", "labeler": PM, "target": {"site": {"evidence": {"atomId": "at-k2"}}},
     "reason": "kept", "note": "The verdict stands: a billing office is not a job site."},
    {"head": "site", "verdict": "same_site", "labeler": PM,
     "target": {"a": {"atomId": "at-k0"}, "b": {"atomId": "at-k3"}}, "note": "Two stores, two addresses."},
]


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def batch(schema):
    deal = DealExample.from_training_blob({"labels": LABELS, "judgments": JUDGMENTS}, ATOMS,
                                          deal_id="synthetic-why2", serve_lines_only=False)
    return featurize(deal, schema)


def _model(schema, cfg=SMALL, seed=0):
    torch.manual_seed(seed)
    return C3Model(schema, cfg)


# ---------------------------------------------------------------- free rows on r

@pytest.mark.parametrize("cfg", [SMALL, ASKED])
def test_an_r_only_row_starts_where_the_plain_row_was_and_ends_reading_no_h(schema, batch, cfg):
    model = _model(schema, cfg)
    plain = model.add_free_head(TYPE)
    with torch.no_grad():
        plain.weight.normal_()
        plain.bias.normal_()
    model.eval()
    with torch.no_grad():
        before = model(batch.oneshot()).logits[TYPE].clone()
    row = model.add_free_head(free_rows.R_ONLY + TYPE)
    assert isinstance(row, free_rows.RFreeRow) and model.free_heads[TYPE] is row
    assert row.weight.shape == (plain.weight.shape[0], cfg.d_head)
    assert torch.equal(row.h_weight, plain.weight[:, cfg.d_head:])
    with torch.no_grad():
        same = model(batch.oneshot()).logits[TYPE]
    assert torch.allclose(before, same, atol=1e-5)          # h_keep 1: exactly the old reading
    free_rows.set_h_keep(model, 0.0)
    x_r, x, h = torch.randn(3, cfg.d_head), torch.randn(3, cfg.d_head), torch.randn(3, cfg.d).requires_grad_()
    a = free_rows.universal_free(row, x_r, x, h)
    (g,) = torch.autograd.grad(a.sum(), h)
    assert float(g.abs().sum()) == 0.0                       # h_keep 0: the row reads no h ...
    assert torch.allclose(a, free_rows.universal_free(row, x_r, torch.randn(3, cfg.d_head), h))  # ... nor the slot
    model.train()
    assert TYPE in model(batch.inputs()).free_logits


def test_an_r_only_row_round_trips_a_checkpoint(schema, batch, tmp_path):
    model = _model(schema)
    model.add_free_head(TYPE)
    row = model.add_free_head(free_rows.R_ONLY + TYPE)
    with torch.no_grad():
        row.weight.normal_()
    free_rows.set_h_keep(model, 0.25)
    sha = digest([asdict(o) for o in schema.opportunities])
    save_checkpoint(tmp_path / "ck.pt", model, step=1, schema_sha256=sha)
    fresh = _model(schema, seed=1)
    load_checkpoint(tmp_path / "ck.pt", fresh, schema_sha256=sha)
    got = fresh.free_heads[TYPE]
    assert isinstance(got, free_rows.RFreeRow) and float(got.h_keep) == 0.25
    assert torch.equal(got.weight, row.weight)


def test_free_h_out_sets_h_keep_and_company_r_grad_scales_the_trunk(schema, batch):
    model = _model(schema)
    model.add_free_head(free_rows.R_ONLY + TYPE)
    c3_loss(model, batch, LossWeights(free_h_out=0.25))
    assert float(model.free_heads[TYPE].h_keep) == 0.75

    def trunk_grad(scale):
        m = _model(schema)
        m.company_r_grad = scale
        out = m(batch.inputs(), company="purtera")
        out.logits["read:co_action"].sum().backward()
        return m.rationale[0].weight.grad.clone(), m.conduct_base.weight.grad.clone()

    t1, c1 = trunk_grad(1.0)
    t2, c2 = trunk_grad(0.1)
    assert torch.allclose(t2, 0.1 * t1, atol=1e-6, rtol=1e-4)
    assert torch.allclose(c1, c2)                             # the company layer learns at the full rate


def test_company_free_rows_read_the_conduct_line(schema, batch):
    model = _model(schema)
    row = model.add_free_head("read:co_action")
    assert row.weight.shape[1] == SMALL.d_head
    model.eval()
    with torch.no_grad():
        base = model(batch.oneshot(), company="purtera").logits["read:co_action"].clone()
        row.bias.fill_(1.0)
        moved = model(batch.oneshot(), company="purtera")
    assert torch.allclose(moved.logits["read:co_action"], base + 1.0, atol=1e-5)
    assert "read:co_action" in company_param_names(model) or any(
        n.startswith("free_heads.read:co_action") for n in company_param_names(model))


# ---------------------------------------------------------------- the text paths alone

def test_described_alone_leaves_the_free_row_out(schema, batch):
    model = _model(schema)
    row = model.add_free_head(free_rows.R_ONLY + TYPE)
    with torch.no_grad():
        row.weight.normal_()
    out = model(batch.inputs())
    loss = why2.described_alone_loss(model, out, batch, LossWeights(described_alone_p=1.0))
    assert float(loss) > 0
    loss.backward()
    assert row.weight.grad is None or float(row.weight.grad.abs().sum()) == 0.0
    assert float(model.heads["content"].parameters().__next__().grad.abs().sum()) > 0


def test_card_notes_reach_their_heads(schema, batch):
    uni, co = why2.card_rows(_model(schema), batch)
    keys = {(i, k) for i, k, _ in uni}
    assert (0, "jdg:site_role") in keys and (3, "jdg:site_role") in keys
    assert (2, "jdg:site_role:reason") in keys                    # the reason-only card's note
    assert all("kept" not in t.split() for _, k, t in uni if k == "jdg:site_role:reason")
    assert any(k == "jdg:site_role:reason" for _, k, _ in batch.reason_notes)
    model = _model(schema)
    w = LossWeights(card_alone=1.0, card_contrast=1.0)
    out = model(batch.inputs(), company="purtera")
    parts = why2.why2_losses(model, out, batch, model.describe(), w)
    assert float(parts["card_alone"]) > 0 and float(parts["card_contrast"]) > 0
    (parts["card_alone"] + parts["card_contrast"]).backward()
    assert float(model.why_proj.weight.grad.abs().sum()) > 0


def test_company_lines_train_only_the_company_layer(schema, batch):
    model = _model(schema)
    out = model(batch.inputs(), company="purtera")
    rows = why2.company_rows(model, batch, why2.card_rows(model, batch)[1])
    assert any("core region" in t for _, _, t in rows) and all("[purtera]" not in t for _, _, t in rows)
    parts = why2.why2_losses(model, out, batch, model.describe(),
                             LossWeights(co_why_contrast=1.0, co_why_alone=1.0))
    assert float(parts["co_why_contrast"]) > 0 and float(parts["co_why_alone"]) > 0
    model.zero_grad()
    parts["co_why_alone"].backward()
    assert float(model.company_why_proj.weight.grad.abs().sum()) > 0
    assert float(model.conduct_head.parameters().__next__().grad.abs().sum()) > 0
    for name, p in model.heads.named_parameters():               # the firewall: no base head moves
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0, name


def test_numbers_and_pair_cards_get_why_alone_terms(schema, batch):
    model = _model(schema)
    out = model(batch.inputs(), company="purtera")
    desc = model.describe()
    num = why2.number_why_alone_loss(model, out, batch, desc)
    assert float(num) > 0
    assert any(j.key == "jdg:site" and j.note for j in batch.judged)
    card = why2.judgment_why_alone_loss(model, out, batch, desc)
    assert float(card) > 0
    card.backward()
    assert float(model.judgment_note.weight.grad.abs().sum()) > 0
    off = why2.why2_losses(model, out, batch, desc, LossWeights(judgments=0.0))
    assert "judgment_why_alone" not in off                       # trains with the cards stage


def test_every_why2_part_is_in_c3_loss(schema, batch):
    model = _model(schema)
    model.add_free_head(free_rows.R_ONLY + TYPE)
    _, parts = c3_loss(model, batch, LossWeights(described_alone_p=1.0))
    for k in ("described_alone", "card_alone", "card_contrast", "co_why_contrast", "co_why_alone",
              "number_why_alone", "judgment_why_alone"):
        assert k in parts and parts[k] > 0, k


def test_why_swap_reads_no_h(schema, batch, monkeypatch):
    from ml.c3 import why_carry

    model = _model(schema)
    model.add_free_head(TYPE)
    with torch.no_grad():
        model.free_heads[TYPE].weight.normal_()
    out = model(batch.inputs())
    pairs = [(0, 1, TYPE, True), (1, 0, TYPE, True)]
    monkeypatch.setattr(why_carry, "swap_pairs", lambda *a, **k: pairs)
    desc = model.describe()
    a = why_carry.why_swap_loss(model, out, batch, desc)
    noisy = replace(out, h=out.h + 100 * torch.randn_like(out.h))
    b = why_carry.why_swap_loss(model, noisy, batch, desc)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------- portability

def test_a_pretend_new_company_refit_leaves_the_base_bit_identical(schema, batch):
    model = _model(schema)
    model.add_free_head("read:co_action")
    for p in model.parameters():
        p.requires_grad_(False)
    parts = [batch, replace(batch, deal_id="synthetic-why2-b"), replace(batch, deal_id="synthetic-why2-c")]
    before = {n: p.clone() for n, p in model.named_parameters() if n in company_param_names(model)}
    res = pretend_new_company(model, parts, batch, hold=1, passes=2, lr=1e-2)
    assert res["base_identical"] and res["params_identical"] and res["outputs_identical"]
    assert res["refit_labels"] > 0 and res["held_co_action_n"] > 0
    moved = [n for n, p in model.named_parameters() if n in before and not torch.equal(p, before[n])]
    assert moved                                              # the company layer did move


def test_admission_purity_flags_a_drop_resting_on_the_company_line():
    blob = {"company": "purtera", "labels": [
        {"label_key": "a", "label_type": "small_talk", "rejected": "true", "note": "[purtera] We skip these."},
        {"label_key": "b", "label_type": "small_talk", "rejected": "true", "note": "A sign-off.\n[purtera] x"},
        {"label_key": "c", "label_type": "task", "reads_set": {"co_action": "reject"},
         "note": "[purtera] We do not quote this."},
    ]}
    assert admission_impurities(blob) == ["a"]


# ---------------------------------------------------------------- the WHY bank

def test_the_why_bank_is_prior_balanced_off_by_default_and_gated(schema, batch, tmp_path, monkeypatch):
    model = _model(schema)
    bank = why_bank.build_bank(model, [batch])
    labels = bank["labels"].tolist()
    assert labels and len(bank["vectors"]) == len(labels)
    mass = {c: float(bank["weights"][bank["labels"] == c].sum()) for c in set(labels)}
    prior = bank["prior"]
    for c, m in mass.items():                                 # each answer votes its corpus share
        assert abs(m - float(prior[c])) < 1e-6
    ck = tmp_path / "ck.pt"
    torch.save(bank, why_bank.bank_path(ck))
    monkeypatch.delenv(why_bank.ENV, raising=False)
    assert why_bank.maybe_attach(model, ck) is None           # off by default
    monkeypatch.setenv(why_bank.ENV, "1")
    assert why_bank.maybe_attach(model, ck) is None           # no passing gate decision: off
    monkeypatch.setenv(why_bank.ENV, "force")
    model.eval()
    with torch.no_grad():
        plain = model(batch.oneshot()).logits[TYPE].clone()
        handle = why_bank.maybe_attach(model, ck)
        mixed = model(batch.oneshot()).logits[TYPE]
    assert handle is not None and not torch.allclose(plain, mixed)
    assert torch.allclose(mixed.exp().sum(-1), torch.ones(len(batch)), atol=1e-4)


def test_the_bank_gate_needs_fewer_flips_and_no_lost_accuracy():
    def items(acc_hits, flips):
        clean = {f"line|{TYPE}|k{i}": ["a" if i < acc_hits else "b", "a", "d"] for i in range(4)}
        lay = {k: [("z" if n < flips else v[0]), v[1], v[2]] for n, (k, v) in enumerate(clean.items())}
        return {"deals": {"x": {"role": "golden", "kinds": {"clean": clean, "layout": lay}}}}

    assert why_bank.gate_decision(items(2, 3), items(2, 1))["enable"]
    assert not why_bank.gate_decision(items(3, 3), items(2, 1))["enable"]   # holdout accuracy fell
    assert not why_bank.gate_decision(items(2, 1), items(3, 1))["enable"]   # flips did not drop


# ---------------------------------------------------------------- the question draft

def test_the_head_question_draft_fits_the_schema_and_names_no_company(schema):
    g = json.loads((REPO / "ml/c3/fixtures/head_questions.json").read_text(encoding="utf-8"))
    g = {k: v for k, v in g.items() if not k.startswith("_")}
    for key in ("col:label_type", "read:site", "jdg:site_role", "col:admission", "col:supplier",
                "read:sow_section", "read:sow_element", "read:task_owner_role"):
        assert g[key]["question"].endswith("?") and g[key]["how_to_decide"]
    s = apply_guidance(load_schema(), g)
    assert not s.scrubbed
    assert "How to decide" in s.by_key()[TYPE].description


def test_runE_sets_every_why2_weight_with_a_ramp():
    text = (REPO / "ml/c3/pod/runE.sh").read_text(encoding="utf-8")
    assert "TYPE_WEIGHT=${TYPE_WEIGHT:-5}" in text and "WHY_SWAP=${WHY_SWAP:-0.1}" in text
    assert "policy_l1=0 " in text or "policy_l1=0\n" in text
    for term in ("described_alone", "card_alone", "card_contrast", "co_why_contrast", "co_why_alone",
                 "number_why_alone", "judgment_why_alone", "free_h_out", "company_r_grad"):
        assert f"{term}=" in text, term
    for key in free_rows.R_ONLY_KEYS:
        assert f"r:{key}" in text or "R_ONLY_KEYS" in text
