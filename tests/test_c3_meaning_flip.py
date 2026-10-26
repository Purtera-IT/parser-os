"""meaning_flip: minimal meaning edits with a certain new answer (ml/c3/meaning_flip.py, losses.meaning_flip_loss)."""
from __future__ import annotations

import ast
import random
import re
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import ml.c3.losses as losses  # noqa: E402
from ml.c3 import meaning_flip as mf  # noqa: E402
from ml.c3.data import IGNORE, DealExample, featurize, number_tokens  # noqa: E402
from ml.c3.loop import SMALL  # noqa: E402
from ml.c3.losses import LossWeights, c3_loss, meaning_flip_loss  # noqa: E402
from ml.c3.model import C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ml" / "c3" / "fixtures" / "synthetic_deal.json"
GATE = Path.home() / "c3-local" / "runE" / "contrast_gate.py"


def _answer(key: str, value: str) -> str:
    opp = load_schema().by_key()[key]
    ix = opp.index(value)
    assert ix is not None, (key, value)
    return opp.answers[ix].description


# ---------------------------------------------------------------- each edit gives the stated answer
def test_the_definitions_the_answers_rest_on():
    """The certainty of every family is read off these definitions; if one changes, re-check the family."""
    assert "provider" in _answer("col:supplier", "us").lower()
    assert "end customer does, supplies" in _answer("col:supplier", "customer").lower()
    assert "number, street, city and zip" in _answer("read:address_level", "full")
    assert "street address and city, but the zip code is missing" in _answer("read:address_level", "street_no_zip")
    assert "explicitly leaves out" in _answer("col:label_type", "exclusion")
    assert "leaves out" in _answer("read:scope_facet", "exclusion")
    assert "carries no fact about the work" in _answer("col:label_type", "small_talk")
    assert "ALSO says something about the job" in _answer("col:label_type", "small_talk")
    assert "teaches no fact" in _answer("col:admission", "drop")
    assert "states a fact of its own about the work" in _answer("col:admission", "keep")


def _edit(fam, text, seed=0):
    return mf.EDITS[fam](text, random.Random(seed))


@pytest.mark.parametrize("seed", range(6))
def test_supplier_swaps_only_the_party_that_does_it(seed):
    new, tg = _edit("sup_to_us", "Customer will provide the rack and patch cables.", seed)
    assert tg == {"col:supplier": "us"} and new.startswith(mf.OUR_SUBJECTS)
    assert new.endswith(" will provide the rack and patch cables.") and not mf.MENTIONS_CUST.search(new)
    new, tg = _edit("sup_to_us", "2. Client to furnish the network cabinet; AP units will be provided by the client.", seed)
    assert tg == {"col:supplier": "us"} and not mf.MENTIONS_CUST.search(new) and new.startswith("2. Our ")
    new, tg = _edit("sup_to_cust", "We are responsible for mounting the displays.", seed)
    assert tg == {"col:supplier": "customer"} and " is responsible for mounting the displays." in new
    assert new.startswith(mf.CLIENT_SUBJECTS) and not mf.MENTIONS_US.search(new)
    new, tg = _edit("sup_to_cust", "All cabling will be supplied by PurTera.", seed)
    assert tg == {"col:supplier": "customer"}
    assert new.startswith("All cabling will be supplied by ") and new[len("All cabling will be supplied by "):-1] in [
        p[:1].lower() + p[1:] for p in mf.CLIENT_SUBJECTS]
    for v in tg.values():
        assert load_schema().by_key()["col:supplier"].index(v) is not None


@pytest.mark.parametrize("fam,text", [
    ("sup_to_us", "Customer will provide floor plans to our team."),             # names us too: not certain
    ("sup_to_us", "Customer will provide access; CDW ships the switches."),      # a third party
    ("sup_to_us", "Customer will not provide parking."),                         # a refusal / exclusion
    ("sup_to_us", "Customer will receive the as-builts."),                       # not the doer
    ("sup_to_cust", "We will provide the cabinet the customer asked for."),      # names the customer too
    ("sup_to_cust", "PurTera will invoice monthly."),                            # a billing term, not a doer
    ("sup_to_cust", "Our engineer will configure the switch."),                  # no subject phrase to swap
    ("addr_street", "123 Main St, Dallas, TX"),                                  # already a street
    ("addr_street", "Danvers | 1000519 | USA-MA Danvers 35 Cherry Hill"),        # a street-like cell
    ("excl_wrap", "Excluded: electrical work"),                                  # already negated
    ("excl_wrap", "Please see the attached drawings."),                          # a pointer, not a work item
    ("excl_wrap", "Site | Task | Hours | 4"),                                    # a table row
])
def test_lines_without_a_certain_answer_are_left_alone(fam, text):
    assert _edit(fam, text) is None


@pytest.mark.parametrize("seed", range(8))
def test_address_street_added_gives_full_or_street_no_zip(seed):
    for text in ("Fontana, CA", "Memphis | 1001823 | PY3 | TN011 | AMEM3 | USA-TN Memphis Pyramid 3",
                 "Site | Bayview | 77002", "Colorno, Italy"):
        new, tg = _edit("addr_street", text, seed)
        level = tg["read:address_level"]
        assert mf.STREETISH.search(new) and any(s in new for s in mf.STREETS)
        has_zip = bool(mf.ZIP.search(new))
        assert level == ("full" if has_zip else "street_no_zip"), (text, new, level)
        assert all(w in new for w in re.findall(r"[A-Za-z]+", text))   # nothing of the line is lost
    new, tg = _edit("addr_street", "Site | Bayview | 77002", seed)
    assert re.match(r"Site \| \d+ [A-Z]\w+ [A-Z]\w+ \| Bayview \| 77002$", new) and tg["read:address_level"] == "full"
    new, tg = _edit("addr_street", "Colorno, Italy", seed)
    assert tg["read:address_level"] == "street_no_zip"   # a ZIP is only added after a US state


@pytest.mark.parametrize("seed", range(4))
def test_exclusion_and_admission_edits(seed):
    new, tg = _edit("excl_wrap", "3) Install 12 access points in the warehouse.", seed)
    assert tg == {"col:label_type": "exclusion", "read:scope_facet": "exclusion"}
    assert new.startswith("3) ") and "Install 12 access points in the warehouse" in new and mf.NEGATION.search(new)
    new, tg = _edit("adm_to_talk", "Terminate 24 Cat6 drops in the IDF.", seed)
    assert tg == {"col:admission": "drop", "col:label_type": "small_talk"} and new in mf.SMALL_TALK
    assert not re.search(r"\d", new)
    new, tg = _edit("adm_add_fact", "Thanks again, have a nice weekend!", seed)
    assert tg == {"col:admission": "keep"} and new.endswith("Thanks again, have a nice weekend!")
    assert any(new.startswith(f) for f in mf.JOB_FACTS) and number_tokens(new)


def test_the_precondition_is_the_saved_gold():
    g = {"col:supplier": "us"}
    assert mf.eligible("sup_to_cust", g) and not mf.eligible("sup_to_us", g)
    assert mf.eligible("excl_wrap", {"col:label_type": "task", "col:admission": "keep"})
    assert not mf.eligible("excl_wrap", {"col:label_type": "task", "col:admission": "drop"})
    assert not mf.eligible("excl_wrap", {"col:label_type": "task", "col:admission": "keep",
                                         "read:scope_facet": "exclusion"})
    assert mf.eligible("adm_add_fact", {"col:label_type": "small_talk", "col:admission": "drop"})
    assert not mf.eligible("adm_add_fact", {"col:label_type": "_keep", "col:admission": "drop"})
    # The same text with another saved answer is never edited.
    texts = ["Customer will provide the rack.", "x", "Customer will provide the rack."]
    flips = mf.plan_flips(texts, {"col:supplier": ["customer", None, "us"]}, random.Random(0))
    assert [f.line for f in flips] == [0]


def test_plan_caps_spreads_and_edits_each_line_once():
    texts = ["Customer will provide the rack."] * 200
    gold = {"col:supplier": ["customer"] * 200, "col:label_type": ["task"] * 200, "col:admission": ["keep"] * 200}
    flips = mf.plan_flips(texts, gold, random.Random(3), cap=10)
    lines = [f.line for f in flips]
    assert len(lines) == len(set(lines)) and all(b - a > 1 for a, b in zip(lines, lines[1:]))
    per = {}
    for f in flips:
        per[f.family] = per.get(f.family, 0) + 1
    assert per and max(per.values()) <= 10
    assert mf.plan_flips(texts, gold, random.Random(3), cap=10) == flips    # seeded: repeats


def test_held_out_deals_are_never_edited():
    for d in ("010038", "010043@planning#c3part0", "000067@pre_quote", "010447.json"):
        assert mf.held_out(d), d
    assert not mf.held_out("010045@planning#c3part1")


# ---------------------------------------------------------------- disjoint from the gate
#: contrast_gate.py's inserted strings, frozen here for hosts without ~/c3-local (the live file is checked too).
GATE_FROZEN = (
    "Thanks so much, hope you had a great weekend!", "Appreciate it, talk soon.",
    "Thank you again for the quick turnaround, really appreciate your help!",
    "Hope you're doing well and staying dry this week.", "Thanks! Have a great rest of your day.",
    "Great catching up yesterday, thanks for your time.", "Sounds good, thank you!",
    "Happy Friday, enjoy the long weekend.",
    "From: Jordan Avery Sent: Tuesday, June 9, 2026 10:14 AM To: Sam Ortiz Subject: RE: follow up",
    "-----Original Message----- From: Casey Morgan Sent: Monday, May 4, 2026 3:02 PM To: Riley Chen",
    "On Wed, Jun 17, 2026 at 2:35 PM Taylor Brooks wrote:",
    "From: Morgan Lee <morgan.lee@example.com> Sent: Thursday, April 2, 2026 9:41 AM Subject: RE: quote",
    "Also, please quote two technicians to mount six TVs at the site next Tuesday.",
    "Also, the job needs 24 new Cat6 drops terminated in the IDF.",
    "Also, we need one technician on site for 8 hours to replace the switch.",
    "Also, the customer wants all 12 access points installed before July 1.",
    "Also, please include labor to rack and cable two servers.",
    "Out of scope: ", "Excluded: ", "Not included: ", "NOT IN SCOPE: ",
    "PurTera", "the customer", "The customer", "TOTAL", "Subtotal", "Grand Total", "Sites", "N/A",
    "Main Street", "Oak Avenue", "Commerce Drive", "Industrial Blvd",
    "in scope", "included", "includes", "include", "by PurTera",
)


def _gate_strings_live() -> set[str]:
    """Every string literal in contrast_gate.py's edit section that the edits insert."""
    src = GATE.read_text(encoding="utf-8")
    edits = src[src.index("# ---------------------------------------------------------------- edits"):
                src.index("# ---------------------------------------------------------------- which lines")]
    tree = ast.parse(edits)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            # regex sources and format pieces are detectors / glue, not inserted text
            if "\\" in v or "(?" in v or "[" in v or v.startswith("^") or len(v.strip()) < 3 or v.strip() in {"str", "drop"}:
                continue
            if re.fullmatch(r"[a-z_]+", v):        # family / key names ("adm_drop", "street")
                continue
            out.add(v)
    return out


def _grams(s: str, n: int = 3) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9']+", s.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _assert_disjoint(gate: set[str]):
    for mine in mf.TEMPLATES:
        m = mine.lower().replace("{body}", " ").strip()
        for g in gate:
            gl = g.lower().strip()
            assert gl not in m and m not in gl, (mine, g)
            assert not (_grams(m) & _grams(gl)), (mine, g, _grams(m) & _grams(gl))


def test_templates_are_disjoint_from_the_contrast_gate():
    _assert_disjoint(set(GATE_FROZEN))
    # No inserted phrase names a party the gate inserts, or the gate's prefixes, streets or totals.
    joined = "\n".join(mf.TEMPLATES).lower()
    for word in ("purtera", "the customer", "out of scope", "excluded:", "not included", "main street",
                 "oak avenue", "commerce drive", "industrial blvd", "also,", "total"):
        assert word not in joined, word


@pytest.mark.skipif(not GATE.exists(), reason="contrast_gate.py lives in ~/c3-local/runE")
def test_templates_are_disjoint_from_the_live_gate_file():
    live = _gate_strings_live()
    assert set(GATE_FROZEN[:21]) <= live, "contrast_gate.py's banks changed: refresh GATE_FROZEN"
    _assert_disjoint(live)


# ---------------------------------------------------------------- the loss
def _batch():
    schema = load_schema()
    b = featurize(DealExample.load(FIXTURE), schema)
    by_key = schema.by_key()
    texts, targets = list(b.texts), {k: list(v) for k, v in b.targets.items()}

    def put(i, text, **gold):
        texts[i] = text
        for key, v in gold.items():
            targets[key][i] = by_key[key].index(v) if v is not None else IGNORE

    put(0, "Thanks so much for the call today, enjoy the weekend!", **{"col:label_type": "small_talk",
                                                                      "col:admission": "drop"})
    put(2, "Customer will provide the rack and patch cables.", **{"col:supplier": "customer"})
    put(4, "Install 12 access points in the warehouse.", **{"col:label_type": "task", "col:admission": "keep",
                                                            "read:scope_facet": None})
    put(13, "Fontana, CA", **{"read:address_level": "city_only"})
    return schema, replace(b, texts=texts, numbers=[number_tokens(t) for t in texts], targets=targets)


class _Ignores:
    """A model that reads every edited line as it read the original: the first pass, whatever the texts."""

    def __init__(self, model, out):
        self.schema, self.out = model.schema, out

    def __call__(self, *a, **kw):
        return self.out


class _Reads(_Ignores):
    """A model that reads the edits: each edited line's logits put on the edit's answer."""

    def __init__(self, model, out, flips):
        super().__init__(model, out)
        self.flips = flips

    def __call__(self, inputs, **kw):
        import copy

        moved = copy.copy(self.out)
        moved.logits = {k: v.clone() for k, v in self.out.logits.items()}
        by_key = self.schema.by_key()
        for f in self.flips:
            for k, v in f.targets.items():
                if k in moved.logits:
                    row = torch.full_like(moved.logits[k][f.line], -20.0)
                    row[by_key[k].index(v)] = 20.0
                    moved.logits[k][f.line] = row
        return moved


def test_the_loss_is_positive_when_the_model_ignores_the_edit():
    schema, batch = _batch()
    torch.manual_seed(0)
    model = C3Model(schema, SMALL).eval()
    w = LossWeights()
    with torch.no_grad():
        desc = model.describe()
        out = model(batch.inputs(), company=batch.company, desc=desc)
    flips = mf.plan_flips(batch.texts, mf.gold_of(batch, schema), random.Random(0))
    assert {f.family for f in flips} >= {"sup_to_us", "excl_wrap", "adm_add_fact", "addr_street"}
    with torch.no_grad():
        ignores = meaning_flip_loss(_Ignores(model, out), out, batch, w, rng=random.Random(0), desc=desc)
        reads = meaning_flip_loss(_Reads(model, out, flips), out, batch, w, rng=random.Random(0), desc=desc)
        real = meaning_flip_loss(model, out, batch, w, rng=random.Random(0), desc=desc)
    assert float(ignores) > 0.5 and float(real) > 0.5
    assert float(reads) < 1e-6 < float(ignores)
    # It trains the edited lines' NEW answers, not the old reading: the ignoring model's first pass already
    # answers the old gold on these lines, and that is what costs.


def test_held_out_parts_and_parts_with_nothing_to_edit_cost_nothing(monkeypatch):
    schema, batch = _batch()
    model = C3Model(schema, SMALL).eval()
    with torch.no_grad():
        out = model(batch.inputs(), company=batch.company)
    calls = []
    monkeypatch.setattr(model, "forward", lambda *a, **k: calls.append(1))
    held = replace(batch, deal_id="010387@planning#c3part0")
    assert float(meaning_flip_loss(model, out, held, LossWeights(), rng=random.Random(0))) == 0.0
    monkeypatch.setattr(mf, "plan_flips", lambda *a, **k: [])
    assert float(meaning_flip_loss(model, out, batch, LossWeights(), rng=random.Random(0))) == 0.0
    assert not calls


def test_meaning_flip_reuses_the_steps_descriptions(monkeypatch):
    schema, batch = _batch()
    model = C3Model(schema, SMALL)
    calls = {"n": 0}
    real = model.describe

    def counted():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(model, "describe", counted)
    _, parts = c3_loss(model, batch, LossWeights(meaning_flip=0.3, entity_swap=0.0, keyword_dropout=0.0))
    assert "meaning_flip" in parts and parts["meaning_flip"] > 0
    assert calls["n"] == 1


def _step(schema, batch, w, state):
    model = C3Model(schema, SMALL)
    model.load_state_dict(state)
    torch.manual_seed(7)
    total, parts = c3_loss(model, batch, w)
    total.backward()
    grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    return float(total.detach()), parts, grads, torch.get_rng_state()


def test_weight_zero_is_a_no_op(monkeypatch):
    schema, batch = _batch()
    torch.manual_seed(0)
    state = {k: v.clone() for k, v in C3Model(schema, SMALL).state_dict().items()}
    off = LossWeights(meaning_flip=0.0)
    t0, p0, g0, r0 = _step(schema, batch, off, state)

    def never(*a, **k):
        raise AssertionError("meaning_flip ran at weight 0")

    # The code path before this term existed: never reached, no draw from torch's generator.
    monkeypatch.setattr(losses, "meaning_flip_loss", never)
    t1, p1, g1, r1 = _step(schema, batch, off, state)
    assert "meaning_flip" not in p0 and t0 == t1 and p0 == p1 and torch.equal(r0, r1)
    assert g0.keys() == g1.keys() and all(torch.equal(g0[k], g1[k]) for k in g0)
    monkeypatch.undo()
    t2, p2, g2, _ = _step(schema, batch, replace(off, meaning_flip=0.3), state)
    assert "meaning_flip" in p2 and p2["meaning_flip"] > 0 and t2 != t0
    assert any(not torch.equal(g0[k], g2[k]) for k in g0)


def test_runE_sets_meaning_flip_ramps_it_and_counts_its_graph():
    text = (ROOT / "ml/c3/pod/runE.sh").read_text(encoding="utf-8")
    assert "MEANING_FLIP=${MEANING_FLIP:-0.3}" in text
    assert 'meaning_flip="$MEANING_FLIP"' in text
    assert re.search(r"for t in [^\n]*\bmeaning_flip\b", text)
    assert '[[ "$MEANING_FLIP" != 0 ]] && graphs=$((graphs + 1))' in text
    assert "|MEANING_FLIP|" in text   # the restart line carries it
