"""The base run's loader, long-deal parts and split guards, on invented rows."""
import pytest

from ml.c3.data import IGNORE, DealExample, featurize
from ml.c3.export import from_page_export
from ml.c3.holdout import LABEL_HOLDOUT, KIT_HOLDOUT
from ml.c3.kit import lock_kit_split
from ml.c3.longdeal import plan_long_deal
from ml.c3.loop import lock_split, lock_validation, part_scales, pool_scores, training_batches
from ml.c3.schema import load_schema

PM = "reviewer@example.com"


def atom(key, text=None, t=0, **kw):
    return {"label_key": key, "atom_id": f"at-{key}", "text": text or f"Line {key}.",
            "entered_at": f"2026-01-01T00:{t:02d}:00Z", **kw}


def label(key, **kw):
    return {"label_key": key, "labeler": PM, "label_type": "task", "reads_set": {}, **kw}


def page_atom(key, art, text, **kw):
    return {"labelKey": key, "atomId": f"at-{key}", "artifactId": art, "text": text,
            "docType": "", "section": ["Scope", "Install"], "saidBy": {"role": "pm", "side": "us"}, **kw}


def test_page_export_becomes_a_training_file_with_arrival_times():
    page = {
        "dealId": "uuid-1", "compileId": "c1", "myLinks": [], "judgments": [], "labels": [],
        "documents": [
            {"artifactId": "mail", "kind": "email", "origin": {"sent_at": "2026-02-01T00:00:00Z"}},
            # A file's authoredAt is not when the deal saw it: it takes the time before it.
            {"artifactId": "file", "kind": "pdf", "authoredAt": "2019-01-01T00:00:00Z"},
        ],
        "atoms": [page_atom("a", "mail", "First."), page_atom("b", "file", "Second."),
                  # Typed on the labeling page: no document, so the deal's latest time.
                  page_atom("c", "art_pm_answers_1", "Third.")],
    }
    doc = from_page_export(page, "010999")
    assert doc["deal_id"] == "010999" and doc["links"] == []
    times = {a["label_key"]: a["entered_at"] for a in doc["atoms"]}
    assert times == {"a": "2026-02-01T00:00:00Z", "b": "2026-02-01T00:00:00Z", "c": "2026-02-01T00:00:00Z"}
    assert doc["atoms"][0]["section"] == "Scope > Install"
    assert doc["atoms"][0]["speaker_role"] == "pm"
    deal = DealExample.from_training_blob(doc, doc["atoms"], deal_id=doc["deal_id"])
    assert [a.key for a in deal.atoms] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="share labelKey"):
        from_page_export({**page, "atoms": page["atoms"] + [page_atom("a", "mail", "Again.")]}, "010999")


def test_dropped_copy_trains_on_its_own_line_not_the_kept_one():
    schema = load_schema()
    blob = {"labels": [label("k")], "judgments": [
        {"head": "suppression", "target_key": "sup:k@line_start=4", "labeler": PM,
         "verdict": "correctly_dropped", "text": "Line k.",
         "target": {"isFold": True, "survivor": "Line k.", "text": "Line k."}},
        {"head": "suppression", "target_key": "sup:k@line_start=9", "labeler": PM,
         "verdict": "should_have_been_kept", "text": "Line k.",
         "target": {"isFold": True, "survivor": "Line k.", "text": "Line k."}},
    ]}
    deal = DealExample.from_training_blob(blob, [atom("k"), atom("z", t=1)], deal_id="synthetic")
    keys = [a.key for a in deal.atoms]
    assert keys[:3] == ["k", "sup:k@line_start=4", "sup:k@line_start=9"]
    batch = featurize(deal, schema)
    adm = schema.by_key()["col:admission"]
    col = batch.targets["col:admission"]
    assert col[0] == adm.index("keep")              # the kept line keeps its own answer
    assert col[1] == adm.index("drop") and col[2] == adm.index("keep")
    assert batch.weights[1] < batch.weights[2]      # the correctly-dropped flood counts less


def test_line_card_wins_over_a_disagreeing_dropped_card_and_is_reported():
    schema = load_schema()
    blob = {"labels": [label("k")], "judgments": [
        {"head": "suppression", "target_key": "sup:k", "labeler": PM, "verdict": "correctly_dropped",
         "text": "Line k.", "target": {"isFold": False, "survivor": None}}]}
    batch = featurize(DealExample.from_training_blob(blob, [atom("k")], deal_id="synthetic"), schema)
    assert batch.targets["col:admission"][0] == schema.by_key()["col:admission"].index("keep")
    assert batch.notes and "disagrees" in batch.notes[0]


def test_a_line_over_several_sites_keeps_every_tier():
    schema = load_schema()
    blob = {"labels": [label("k", reads_set={"location_tier": ["major_metro", "rural"]})]}
    batch = featurize(DealExample.from_training_blob(blob, [atom("k")], deal_id="synthetic"), schema)
    opp = schema.by_key()["read:location_tier"]
    assert batch.multi_targets[opp.key][0] == tuple(sorted({opp.index("major_metro"), opp.index("rural")}))
    assert batch.context["location_tier"][0] == "major_metro+rural"


def _long_deal():
    atoms = [atom(f"a{i}", t=i) for i in range(12)]
    blob = {
        "labels": [label(f"a{i}") for i in range(12)],
        "links": [{"from_key": "a1", "to_label_key": "a10", "relation": "supports", "labeler": PM},
                  {"from_key": "a2", "to_label_key": "a3", "relation": "supports", "labeler": PM}],
        "judgments": [{"head": "billing_type", "target_key": "deal", "labeler": PM, "verdict": "fixed"}],
    }
    return DealExample.from_training_blob(blob, atoms, deal_id="010999")


def test_long_deal_teaches_each_label_once_and_packs_far_links():
    schema = load_schema()
    deal = _long_deal()
    plan = plan_long_deal(deal, schema, max_atoms=5, overlap=1, context=1)
    assert len(plan.windows) == 3 and len(plan.bridges) == 1 and len(plan.digest) == 1
    assert sum(sum(b.labeled) for b in plan.parts) == 12         # once each, never in the overlap twice
    assert sum(sum(b.labeled) for b in plan.bridges + plan.digest) == 0
    pack = plan.bridges[0]
    assert len(pack) <= 5 and pack.edges["supports"]             # a1 -> a10 trains in the pack
    home = plan.windows[0]
    assert home.link_elsewhere["supports"] == [1]                # not taught "no link" at home
    assert any(j.key == "jdg:billing_type" for j in plan.digest[0].judged)
    assert not any(j.size == "deal" for b in plan.windows for j in b.judged)
    assert not plan.digest[0].edges                              # links train once, not in the digest
    parts, notes = training_batches(deal, schema, max_atoms=5)
    assert {p.deal_id.split("#")[0] for p in parts} == {"010999"}
    assert any("digest" in n for n in notes)


def test_fixed_holdouts_never_train_in_either_run():
    for deal_id in LABEL_HOLDOUT + KIT_HOLDOUT:
        with pytest.raises(ValueError, match="[Hh]oldout"):
            lock_kit_split([deal_id])
    train = [featurize(DealExample.from_training_blob({}, [atom("a")], deal_id=f"{LABEL_HOLDOUT[0]}#c3part0"),
                       load_schema())]
    other = [featurize(DealExample.from_training_blob({}, [atom("a")], deal_id="010998"), load_schema())]
    with pytest.raises(ValueError, match="fixed holdout"):
        lock_split(train, other)
    with pytest.raises(ValueError, match="Validation deal"):
        lock_validation(other, train, other)


def test_long_deals_are_balanced_and_scores_pool_by_count():
    schema = load_schema()
    parts, _ = training_batches(_long_deal(), schema, max_atoms=5)
    short = featurize(DealExample.from_training_blob({}, [atom("a")], deal_id="010998"), schema)
    scale = part_scales(parts + [short])
    assert scale["010998"] == 1.0
    assert scale[parts[0].deal_id] == pytest.approx(len(parts) ** -0.5)
    pooled = pool_scores([{"h": {"exact": 1.0, "n": 1.0}}, {"h": {"exact": 0.0, "n": 3.0}}])
    assert pooled["h"] == {"exact": 0.25, "n": 4.0}
    assert IGNORE < 0


def test_window_size_is_a_setting_and_released_encoder_blocks_learn_slower(tmp_path):
    import json

    from ml.c3.loop import SMALL, _student_optimizer, load_split, release_encoder_top
    from ml.c3.model import C3Model

    schema = load_schema()
    deal = _long_deal()
    path = tmp_path / "deal.json"
    path.write_text(json.dumps({"deal_id": deal.deal_id, "atoms": [atom(f"a{i}", t=i) for i in range(12)],
                                "labels": [label(f"a{i}") for i in range(12)], "links": [], "judgments": []}))
    assert len(load_split([str(path)], schema, max_atoms=5)[0]) > 1
    assert len(load_split([str(path)], schema, max_atoms=64)[0]) == 1
    model = C3Model(schema, SMALL)
    assert release_encoder_top(model, 2) == 0            # a hashed encoder has no blocks to release
    opt = _student_optimizer(model, 1e-3, True, 1e-5)
    assert sorted(g["lr"] for g in opt.param_groups) == [1e-5, 1e-3]


def test_a_link_from_a_rule_card_finds_its_line_through_the_card():
    long = "The invoiced amount will equal the fees for each completed milestone, as listed in the table below."
    atoms = [atom("r", long), atom("r2", long, t=1, doc_id="other"), atom("n", "Tasks outside this scope.", t=2)]
    atoms[0]["doc_id"] = atoms[2]["doc_id"] = "sow"
    blob = {
        "judgments": [{"head": "rule", "target_key": "rule:lead_in:1", "labeler": PM, "text": long,
                       "verdict": "should_not_fire"}],
        # The page cut the card's text off mid-word.
        "links": [{"from_head": "rule", "from_key": "rule:lead_in:1", "from_text": long[:60],
                   "to_label_key": "n", "relation": "near_miss", "labeler": PM}],
    }
    deal = DealExample.from_training_blob(blob, atoms, deal_id="synthetic")
    assert deal.edges == [("r", "n", "near_miss")]       # the copy in the linked file, not the other one


def test_the_base_writes_one_file_per_deal_and_scores_the_holdout_once(tmp_path):
    from ml.c3.gates import save_checkpoint
    from ml.c3.loop import SMALL
    from ml.c3.model import C3Model
    from ml.c3.predict import read_deal
    from ml.c3.readiness import digest
    from dataclasses import asdict

    schema = load_schema()
    model = C3Model(schema, SMALL)
    deal = _long_deal()
    doc = read_deal(model, deal, schema, max_atoms=5)
    bare = DealExample.from_training_blob({}, [atom(f"b{i}", t=i) for i in range(3)], deal_id="010997")
    assert set(read_deal(model, bare, schema)["lines"]) == {"b0", "b1", "b2"}   # an unlabeled deal reads too
    assert set(doc["lines"]) == {f"a{i}" for i in range(12)}          # every line, once
    assert doc["lines"]["a0"]["text"] == "Line a0."                     # the words travel: prompts read them
    reading = doc["lines"]["a0"]["reads"]["col:label_type"]
    assert reading["answer"] in [a.value for a in schema.by_key()["col:label_type"].answers]
    assert "jdg:billing_type" in doc["deal"]
    assert all({"src", "dst", "relation", "p"} <= set(link) for link in doc["links"])
    path = tmp_path / "run.pt"
    save_checkpoint(str(path), model, step=1, schema_sha256=digest([asdict(o) for o in schema.opportunities]))
    assert path.exists()


def test_a_deal_kit_file_is_rebuilt_on_the_base_read():
    from ml.c3.kit import KitExample
    from ml.c3.predict import kit_from_read

    kit = {"deal_id": "010996", "atoms": [{"key": "s", "text": "Site A.", "type": "physical_site"},
                                          {"key": "t", "text": "Mount cameras.", "type": "task"}],
           "lines": [{"section": "sites", "atom_key": "s"}, {"section": "labor", "atom_key": "t", "hours": 4}]}
    read = {"lines": {"s": {"reads": {"col:label_type": {"answer": "physical_site", "p": 0.9}}},
                      "t": {"reads": {"col:label_type": {"answer": "deliverable", "p": 0.5}}}},
            "links": [{"src": "t", "dst": "s", "relation": "supports", "p": 0.8}]}
    out, stats = kit_from_read(kit, read)
    assert stats == {"atoms": 2, "retyped": 1, "not_read": 0, "lines_unhosted": 1}
    ex = KitExample.from_dict(out)                    # loads: no line points at a mistyped atom
    assert [line.atom_key for line in ex.lines] == ["s", None]
    assert ex.links == (("t", "s", "supports"),)


def test_odd_saved_values_teach_what_they_can_and_are_reported():
    schema = load_schema()
    blob = {"labels": [label("h", reads_set={"labor_hours": "4-6"}),
                       label("d", reads_set={"display_size": "55in"}),
                       label("n", reads_set={"needed_by": ["2026-07-27", "sow"]})]}
    batch = featurize(DealExample.from_training_blob(
        blob, [atom("h"), atom("d", t=1), atom("n", t=2)], deal_id="synthetic"), schema)
    assert batch.numbers_target["read:labor_hours"][0] == 5.0       # the middle of the range
    assert batch.numbers_target["read:display_size"][1] == 55.0     # the leading number
    needed = schema.by_key()["read:needed_by"]
    assert batch.multi_targets[needed.key][2] == (needed.index("sow"),)
    assert any("read:needed_by on n" in note for note in batch.notes)


def test_the_teacher_reads_only_lines_with_something_to_teach():
    from ml.c3.brain import told_why
    from ml.c3.losses import teacher_rows

    schema = load_schema()
    blob = {"labels": [label("a1"), label("a3")]}
    batch = featurize(DealExample.from_training_blob(
        blob, [atom(f"a{i}", t=i) for i in range(4)], deal_id="synthetic"), schema)
    rows, taught = teacher_rows(batch, told_why(batch, schema))
    assert rows == [1, 3]
    assert len(taught) == 2 and taught.targets["col:label_type"] == [batch.targets["col:label_type"][i] for i in rows]


def test_same_document_masks_are_tensors_built_without_a_python_square():
    schema = load_schema()
    atoms = [atom("a", doc_id="x", section="S"), atom("b", t=1, doc_id="x", section="T"),
             atom("c", t=2, doc_id="", section="S")]
    batch = featurize(DealExample.from_training_blob({"labels": []}, atoms, deal_id="synthetic"), schema)
    assert batch.same_doc.tolist() == [[True, True, False], [True, True, False], [False, False, False]]
    assert batch.same_section.tolist() == [[True, False, False], [False, True, False], [False, False, True]]


def test_a_field_note_the_registry_does_not_list_still_explains_its_field():
    schema = load_schema()
    blob = {"labels": [label("r", reads_set={"rate": "present", "rate_note": "The rate card sets the hourly rate."})]}
    batch = featurize(DealExample.from_training_blob(blob, [atom("r")], deal_id="synthetic"), schema)
    assert "read:rate" in batch.field_notes[0]


def test_a_bf16_encoder_reads_a_deal_on_the_cpu():
    """The live encoder is bf16 under float32 layers; a CPU read must not
    trip on the mix, and causal links must not need float64 on the device."""
    from ml.c3.loop import SMALL
    from ml.c3.model import C3Model
    from ml.c3.predict import read_deal
    from ml.c3.text import HashingEncoder

    class Bf16(HashingEncoder):
        def forward(self, texts):
            return super().forward(texts).bfloat16()

    schema = load_schema()
    model = C3Model(schema, SMALL, text=Bf16(SMALL.d_text))
    doc = read_deal(model, _long_deal(), schema, max_atoms=5, device="cpu")
    assert set(doc["lines"]) == {f"a{i}" for i in range(12)}


def test_a_lines_only_read_skips_links_and_deal_and_honours_its_deadline():
    import time

    from ml.c3.loop import SMALL
    from ml.c3.model import C3Model
    from ml.c3.predict import read_deal

    schema = load_schema()
    model = C3Model(schema, SMALL)
    deal = _long_deal()
    doc = read_deal(model, deal, schema, max_atoms=5, lines_only=True)
    assert set(doc["lines"]) == {f"a{i}" for i in range(12)}
    assert doc["links"] == [] and doc["deal"] == {}
    with pytest.raises(TimeoutError):
        read_deal(model, deal, schema, max_atoms=5, lines_only=True, deadline=time.monotonic() - 1)


def test_a_bf16_copy_reads_every_line_on_the_cpu():
    import torch

    from ml.c3.loop import SMALL
    from ml.c3.model import C3Model
    from ml.c3.predict import read_deal

    schema = load_schema()
    model = C3Model(schema, SMALL).to(torch.bfloat16)
    doc = read_deal(model, _long_deal(), schema, max_atoms=5, lines_only=True)
    assert set(doc["lines"]) == {f"a{i}" for i in range(12)}
    assert all(0.0 <= r["reads"]["col:label_type"]["p"] <= 1.0 for r in doc["lines"].values())
