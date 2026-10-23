"""The signal audit (ml/c3/signal_audit.py) on an invented deal.

* the census finds each signal the loader would train, and ablate removes it;
* a signal the student's losses read is OK, the [parser] line is UNUSED;
* the gate fails on a counted signal that trains nothing, unless allowlisted;
* the loss table is read from pod/runE.sh as the script resolves it.
"""
from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("torch")

from ml.c3 import signal_audit as sa  # noqa: E402

PM = "Pat Morgan"


def _atom(k, text, t):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": "email-1", "doc_kind": "email",
            "entered_at": f"2026-05-01T10:{t:02d}:00Z", "speaker_role": "customer", "speaker_side": "customer"}


def _payload():
    texts = ["Install window is Saturdays.", "Saturday installs confirmed by the site lead.",
             "Two techs for four hours.", "Thanks, talk soon."]
    atoms = [_atom(f"k{n}", t, n) for n, t in enumerate(texts, 1)]
    labels = [
        {"label_key": "k1", "label_type": "schedule", "labeler": PM,
         "note": "The work happens on Saturdays.\n[purtera] keep: we staff weekends at the weekend rate.\n"
                 "[parser] split from the line above."},
        {"label_key": "k2", "label_type": "schedule", "labeler": PM, "note": "Confirms the window."},
        {"label_key": "k3", "label_type": "labor", "labeler": PM, "note": "2 techs x 4 hours = 8 hours."},
        {"label_key": "k4", "label_type": "_keep", "labeler": PM, "note": "A sign-off.",
         "reads_set": {"noise_class": "salutation"}},
    ]
    links = [{"from_head": "atom", "from_key": "k2", "to_label_key": "k1", "relation": "supports",
              "note": "Both lines name the same install window.", "labeler": PM}]
    return {"deal_id": "900001", "company": "purtera", "atoms": atoms, "labels": labels,
            "judgments": [], "links": links, "deal_answers": []}


def test_census_and_ablate():
    p = _payload()
    c, w = sa.census(p)
    assert w["note.purtera"] == 8 and w["link_note.supports"] == 7
    assert c["note.universal"] == 4 and c["note.purtera"] == 1 and c["note.parser"] == 1
    assert c["note.arithmetic"] == 1 and c["link.supports"] == 1 and c["link_note.supports"] == 1
    assert c["label.label_type"] == 4 and c["read.noise_class"] == 1
    q = copy.deepcopy(p)
    sa.ablate(q, "note.purtera")
    assert "[purtera]" not in q["labels"][0]["note"] and "[parser]" in q["labels"][0]["note"]
    sa.ablate(q, "link_note.supports")
    assert q["links"][0]["note"] == "" and sa.census(q)[0]["link.supports"] == 1
    sa.ablate(q, "read.noise_class")
    assert "noise_class" not in q["labels"][3]["reads_set"]


def test_audit_statuses_and_gate(tmp_path):
    f = tmp_path / "900001.json"
    f.write_text(json.dumps(_payload()))
    out = tmp_path / "a.json"
    base = ["--tiny", "--train", str(f), "--json", str(out), "--out", str(tmp_path / "a.md"),
            "--only", r"^(note\.(parser|universal)|link_note\.supports|label\.label_type)$",
            "--stage-steps", "lines=1,links=1,cards=1,numbers=1,deal=0,open_links=1"]
    assert sa.main(base) == 0
    found = json.loads(out.read_text())["findings"][tmp_path.name]
    assert found["note.parser"]["status"] == "UNUSED"
    for s in ("note.universal", "link_note.supports", "label.label_type"):
        assert found[s]["status"] == "OK", (s, found[s])
    assert found["link_note.supports"]["terms"]
    assert sa.main(base + ["--min-count", "1"]) == 1
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"note.parser": "parse remarks, dropped by the note grammar"}))
    assert sa.main(base + ["--min-count", "1", "--allowlist", str(allow)]) == 0
    cov = {r["kind"]: r for r in json.loads(out.read_text())["coverage"]}
    assert cov["label WHY (universal)"]["covered"] == cov["label WHY (universal)"]["words"] > 0
    assert cov["description"]["covered"] > 0
    # Reasoning text that reaches no loss fails the launch: the [purtera] line here.
    assert sa.main(base + ["--only", r"^note\.(universal|purtera)$", "--min-kind-words", "5"]) == 1


def test_weights_from_run_script():
    loss, heads, stages = sa.weights_from_script(str(sa.REPO / "ml/c3/pod/runE.sh"))
    assert loss["why_contrast"] == 0.3 and loss["why_align"] == 0.0 and loss["link_why_contrast"] == 0.3
    assert heads == {"col:label_type": 5.0}
    # The why2 block is part of the table; its knobs are not terms.
    assert loss["card_alone"] == 0.2 and loss["co_why_contrast"] == 0.2 and loss["company_r_grad"] == 0.1
    assert stages.startswith("lines=12,links=6,cards=6")
    union, first = sa.run_weights(loss, heads, stages)
    assert union.judgments == 1.0 and first["judgments"] == "cards" and first["relations"] == "links"
    assert union.why_align == 0.0 and union.deal_numbers == 0.2
    assert union.card_contrast == 0.2 and "free_h_out" not in first and "card_cap" not in first


def test_bypass_and_portability():
    """The h-path share sees a free head that reads h; a company refit step
    leaves every universal logit bit-identical and moves a company one."""
    import torch

    from ml.c3 import smoke_checks as sc
    from ml.c3.data import DealExample
    from ml.c3.loop import SMALL, training_batches
    from ml.c3.model import C3Model
    from ml.c3.schema import load_schema

    schema = load_schema()
    p = _payload()
    deal = DealExample.from_training_blob(p, p["atoms"], deal_id=p["deal_id"])
    batch = training_batches(deal, schema, max_atoms=64)[0][0]
    torch.manual_seed(0)
    model = C3Model(schema, SMALL)
    model.add_free_head("col:label_type")
    assert sc.h_path_shares(model, batch, ("col:label_type",))["col:label_type"] == 0.0
    with torch.no_grad():
        model.free_heads["col:label_type"].weight.normal_()
    assert sc.h_path_shares(model, batch, ("col:label_type",))["col:label_type"] > 0.10
    port = sc.portability(model, batch)
    assert port["repeatable"] and not port["base_changed"] and port["company_moved"]
