"""Generated row layouts (ml/c3/layout_why.py) never reach the swap gate's held-out families.

* every output of swap_gate.py's own ``layout`` (kv_lines, markdown, semicolon) and ``layout_new`` (html,
  fixed, sentence) on these rows is recognised by ``gate_shape`` (the detector is not blind);
* thousands of ``relayout`` draws over the same rows are never recognised, and never equal a gate output;
* generated field names (``synonym``, the broad keyword-dropout mask) never use a word of swap_gate's FIELD_SYN
  or SURFACE values; the copies of those values here match swap_gate.py;
* the loss is a named part of c3_loss when on, absent at weight 0, and ties the WHY.
"""
from __future__ import annotations

import ast
import hashlib
import os
import random
import re
from dataclasses import replace
from pathlib import Path

import pytest

from ml.c3 import layout_why as lw
from ml.c3.why_carry import FIELD_SYNONYMS, _FIELD, mask_surface

SWAP_GATE = Path(os.environ.get("C3_SWAP_GATE", Path.home() / "c3-local/runE/swap_gate.py"))

ROWS = [
    "Site ID: 1042 | Address: 123 Main St | City: CONROE | State: TX | Zip: 77304 | Total Hrs: 6.5",
    "City: CONROE | Zip: 77304 | Floors: 2 | Total Hrs: 6.5",
    "Kroger #212 | 9800 Westheimer Rd | Houston | TX | 77042",
    "Kroger #212\tHouston\tTX\t4 floors\t12 drops",
    "Store #: 0451 | Name: Walgreens Midtown | Phone: 713-555-0101 | Contact: Dana Ruiz | Email: dana@example.com",
    "Qty: 4 | Description: 55 inch display wall mount | Hours: 2.5 | Region: South",
    "Site: Lakeside Plaza | 450 Commerce Blvd, Suite 210 | Bldg SqFt: 120000 | Notes: after 6pm only",
    "Location: DC-4 | Address: 1 Data Way | Region: West | Sq Ft: 42000 | Floors: 3",
    "Item 7 | Cat6 drop | 48 | each | Provider",
    "Site Name: Mercy Clinic and Lab | Address: 77 Elm St | City: Akron | State: OH",
]


def _gate_functions():
    """swap_gate.py's ``_h``, ``is_row``, ``layout`` and ``layout_new``, read from its source (importing it would
    pull the serving code), plus its FIELD_SYN and SURFACE dicts."""
    if not SWAP_GATE.exists():
        pytest.skip(f"no {SWAP_GATE}")
    tree = ast.parse(SWAP_GATE.read_text())
    want = {"_h", "is_row", "layout", "layout_new"}
    src, consts = [], {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            src.append(ast.get_source_segment(SWAP_GATE.read_text(), node))
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("FIELD_SYN", "SURFACE")
                                                for t in node.targets):
            consts[node.targets[0].id] = ast.literal_eval(node.value)
    ns = {"hashlib": hashlib, "re": re, "cells_of": lw.cells_of}
    exec("\n\n".join(src), ns)  # noqa: S102 (the gate's own code, read from disk)
    assert want <= set(ns), "swap_gate.py no longer defines " + ", ".join(sorted(want - set(ns)))
    return ns, consts


def _variants(row):
    """The row plus variants that hit every gate family (its hash picks one family per text)."""
    out = [row]
    for k in range(40):
        out.append(row.replace(row.split("|")[0].split("\t")[0].strip(),
                               row.split("|")[0].split("\t")[0].strip() + f" {k}", 1))
    return out


def test_gate_copies_match_swap_gate():
    _, consts = _gate_functions()
    assert set(lw.GATE_FIELD_SYN_VALUES) == set(consts["FIELD_SYN"].values())
    assert set(lw.GATE_SURFACE_VALUES) == set(consts["SURFACE"].values())


def test_detector_recognises_every_gate_family():
    ns, _ = _gate_functions()
    seen = set()
    for row in ROWS:
        for v in _variants(row):
            assert ns["is_row"](v) == lw.is_row(v)
            for fn, fams in ((ns["layout"], ("kv_lines", "markdown", "semicolon")),
                             (ns["layout_new"], ("html", "fixed", "sentence"))):
                g = fn(v)
                if g == v:
                    continue
                shape = lw.gate_shape(g, v)
                assert shape in fams, (shape, g)
                seen.add(shape)
    assert seen == {"kv_lines", "markdown", "semicolon", "html", "fixed", "sentence"}


@pytest.mark.parametrize("seed", range(4))
def test_relayout_never_emits_a_gate_family(seed):
    ns, _ = _gate_functions()
    rng = random.Random(seed)
    fams = {}
    for row in ROWS:
        held = {" ".join(ns["layout"](v).split()) for v in _variants(row)} \
            | {" ".join(ns["layout_new"](v).split()) for v in _variants(row)}
        for _ in range(600):
            out = lw.relayout(row, rng)
            if out is None:
                continue
            assert lw.gate_shape(out, row) is None, out
            assert " ".join(out.split()) not in held, out
            assert "<td" not in out.lower() and "<tr" not in out.lower()
            assert not any(re.match(r"^\s*\|?\s*:?-{2,}", ln) for ln in out.split("\n"))
            assert ";" not in out and " = " not in out
            assert not re.search(r"\b(is|and)\b", out) or re.search(r"\b(is|and)\b", row)
            fams[out[:3]] = fams.get(out[:3], 0) + 1
            # values survive verbatim
            assert all(v in out for _, v in lw.cells_of(row))
    assert len(fams) > 50   # many shapes, not a handful


def test_every_family_renders_on_keyed_and_bare_rows():
    rng = random.Random(7)
    for row in (ROWS[0], ROWS[2], ROWS[3]):
        for fam in lw.FAMILIES:
            outs = {lw.relayout(row, rng, family=fam) for _ in range(20)} - {None}
            assert outs or (fam == "spaced" and not all(k for k, _ in lw.cells_of(row))), (fam, row)


def test_generated_names_never_use_a_gate_word():
    _, consts = _gate_functions()
    gate_values = {v.lower() for v in list(consts["FIELD_SYN"].values()) + list(consts["SURFACE"].values())}
    gate_words = {t for v in gate_values for t in re.findall(r"[a-z0-9]+", v)}
    rng = random.Random(3)
    keys = list(consts["FIELD_SYN"]) + list(lw._SYN_POOL) + ["Site Key", "Total Hrs", "Bldg SqFt", "PO #"]
    names = set()
    for k in keys:
        for n in range(200):
            s = lw.synonym(k, rng, n % 9 + 1)
            names.add(s)
            assert s.lower() not in gate_values, (k, s)
            new = set(re.findall(r"[a-z0-9]+", s.lower())) - set(re.findall(r"[a-z0-9]+", k.lower()))
            assert not new & gate_words, (k, s)
    assert len(names) > 300
    # The masks keyword_dropout already trained: whole names disjoint from the gate's.
    assert not {v.lower() for v in FIELD_SYNONYMS.values()} & gate_values


def test_broad_mask_removes_or_renames_every_key_and_off_is_unchanged():
    _, consts = _gate_functions()
    gate_values = {v.lower() for v in list(consts["FIELD_SYN"].values()) + list(consts["SURFACE"].values())}
    row = "City: CONROE | Zip: 77304 | Address: 1 Elm St | Total Hrs: 6.5"
    a = [mask_surface(row, random.Random(s), word_drop=0.0) for s in range(50)]
    b = [mask_surface(row, random.Random(s), word_drop=0.0, broad=0.0) for s in range(50)]
    assert a == b
    rng = random.Random(1)
    outs = [mask_surface(row, rng, word_drop=0.0, broad=1.0) for _ in range(300)]
    assert any(o and ":" not in o for o in outs)                     # every key removed
    labels = {m.group(2).strip().lower() for o in outs if o for m in _FIELD.finditer(o)}
    assert len(labels) > 40 and not labels & gate_values
    assert all(o and "77304" in o and "CONROE" in o for o in outs)





def test_layout_why_is_a_part_and_ties_the_why():
    torch = pytest.importorskip("torch")
    from ml.c3.data import DealExample, featurize
    from ml.c3.losses import LossWeights, c3_loss
    from ml.c3.model import C3Config, C3Model
    from ml.c3.schema import load_schema

    small = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                     residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)

    def atom(k, text, t):
        return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": "sow-1", "doc_kind": "sow",
                "entered_at": f"2026-05-01T10:{t:02d}:00Z", "speaker_role": "customer", "speaker_side": "customer"}

    atoms = [atom(f"k{i}", r, i) for i, r in enumerate(ROWS[:6])] + [atom("k9", "Work window is weekdays.", 9)]
    labels = [{"label_key": f"k{i}", "label_type": "physical_site",
               "note": f"Building {i} on the customer's list of places to survey."} for i in range(6)]
    labels.append({"label_key": "k9", "label_type": "site_access_window", "note": "When the crew may work."})
    schema = load_schema()
    batch = featurize(DealExample.from_training_blob({"labels": labels}, atoms, deal_id="synthetic-layout-why"),
                      schema)
    torch.manual_seed(0)
    model = C3Model(schema, small)
    out = model(batch.inputs())
    w = replace(LossWeights(), layout_why=0.3, layout_why_p=1.0)
    loss = lw.layout_why_loss(model, out, batch, w, rng=random.Random(0))
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.why_proj.parameters())
    model.zero_grad()
    base = LossWeights(clause_use=0.0, keyword_dropout=0.0, entity_swap=0.0, meaning_flip=0.0)
    _, parts = c3_loss(model, batch, replace(base, layout_why=0.3, layout_why_p=1.0))
    assert "layout_why" in parts and parts["layout_why"] > 0
    _, parts = c3_loss(model, batch, base)
    assert "layout_why" not in parts
