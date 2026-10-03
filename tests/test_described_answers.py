"""Every question and answer the heads are built from has written words.

The C3 heads build each answer's output from its description (ml/c3/schema.py),
and the asked layers seed each question space from its questions' descriptions.
A bare value name teaches nothing, and a base description that names a
company's tools would be cut by the scrub and leave the head thinner.
"""
import json
from pathlib import Path

from ml.c3.schema import dropped, load_schema

ROOT = Path(__file__).resolve().parents[1]
REG = json.loads((ROOT / "app/core/atom_types.json").read_text(encoding="utf-8"))
TRAINED = ("universal", "company")


def _closed(r):
    return "|" in str(r.get("values", ""))


def test_every_closed_answer_has_its_own_description():
    for r in REG["reads"]:
        if r.get("layer") not in TRAINED or not _closed(r):
            continue
        values = [v.strip() for v in r["values"].split("|") if v.strip()]
        docs = r.get("value_desc") or {}
        assert list(docs) == values, f"{r['key']}: value_desc must describe {values} in order"
        for v, d in docs.items():
            assert len(d.split()) >= 5, f"{r['key']}.{v}: too short to teach a head: {d!r}"


def test_every_relation_and_option_is_described():
    for r in REG["relations"]:
        assert r.get("desc"), f"relation {r['key']} has no description"
    for group in ("about", "wants", "suppliers", "context_hints"):
        for o in REG[group]:
            assert o.get("desc"), f"{group}.{o['key']} has no description"


def test_base_descriptions_name_no_company_tools():
    texts = []
    for r in REG["reads"]:
        if r.get("layer") == "universal":
            texts += [(r["key"], r.get("desc", ""))]
            texts += [(f"{r['key']}.{v}", d) for v, d in (r.get("value_desc") or {}).items()]
    texts += [(f"rel:{r['key']}", r.get("desc", "")) for r in REG["relations"]]
    texts += [(f"hint:{o['key']}", o["desc"]) for o in REG["context_hints"]]
    texts += [(f"supplier:{o['key']}", o["desc"]) for o in REG["suppliers"]]
    bad = [(k, s) for k, t in texts for s in dropped(t)]
    assert not bad, f"base text naming a company's tools (move it to a company reading): {bad}"


def test_the_heads_read_the_answer_descriptions():
    by_key = load_schema().by_key()
    overview = {a.value: a.description for a in by_key["read:sow_section"].answers}["overview"]
    assert "executive summary" in overview
    assert "rel:near_miss" in by_key and "answers differently" in by_key["rel:near_miss"].description
