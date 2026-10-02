"""JSON atoms carry their position in the file (live Ox 010353, INTAKE_REQUEST_OX-0038.json).

The atom text is a flattened key path -- ``equipment.items[1].quantity: 1`` --
which never appears in the file, so the source pane could not find it by text,
and with no position on the locator the labeling walk had nothing to order the
atoms by. Every value atom now carries line_start/line_end and char offsets of
its OWN occurrence, measured against the file text the viewer shows.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.ids import stable_id
from app.parsers.json_parser import JsonParser

FIXTURE = {
    "site": {
        "name": "VC Links",
        "address": {"city": "Findlay", "state": "OH", "street": "15733 US-224"},
        "height_requirement": "30 ft",
    },
    "access": {
        "notes": 'He said "call on arrival" \\ then wait',
        "lift_required": "yes",
        "escort_required": False,
        "coi_required": None,
    },
    "equipment": {
        "items": [
            {"sku": "PTZ-2MP", "quantity": 1, "desc": "Sapphire \"PTZ\" camera"},
            {"sku": "SOLAR-400", "quantity": 1, "desc": "400 W solar"},
            {"sku": "MAST-30", "quantity": 1, "tags": ["a", "a", {"deep": {"x": 1}}]},
        ]
    },
    "contacts": [
        {"name": "lisa deviney", "role": "requester", "phone": None},
        {"name": "John Ozuna-Diaz", "role": "csm"},
    ],
    "ünïcødé": {"emoji": "📷 cam", "x/y~z": 2.5e3},
}


def _parse(tmp_path: Path, raw: str, name: str = "INTAKE_REQUEST_OX-0038.json"):
    p = tmp_path / name
    p.write_bytes(raw.encode("utf-8"))
    out = JsonParser().parse_artifact_full(
        project_id="T", artifact_id=stable_id("art", str(p)), path=p,
    )
    return out.atoms


def _resolve(data, pointer: str):
    node = data
    for seg in pointer.split("/")[1:]:
        seg = seg.replace("~1", "/").replace("~0", "~")
        node = node[int(seg)] if isinstance(node, list) else node[seg]
    return node


def _check(text: str, atoms) -> None:
    lines = text.split("\n")
    assert atoms
    for a in atoms:
        loc = a.source_refs[0].locator
        for k in ("line_start", "line_end", "char_start", "char_end", "member_char_start"):
            assert k in loc, (a.raw_text, loc)
        # The span is that value, and only that value.
        frag = text[loc["char_start"]:loc["char_end"]]
        assert json.loads(frag) == _resolve(json.loads(text), loc["json_pointer"]), (a.raw_text, frag)
        # The line it names holds the value.
        line_text = "\n".join(lines[loc["line_start"] - 1:loc["line_end"]])
        assert frag.split("\n")[0] in line_text
        # The member begins at its key (or at the value, for an array element).
        head = text[loc["member_char_start"]:loc["char_start"]]
        last = loc["json_pointer"].rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
        if head:
            assert json.loads(head.rstrip().rstrip(":")) == last
    # Document order: positions strictly increase in emission order.
    starts = [a.source_refs[0].locator["char_start"] for a in atoms]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    line_starts = [a.source_refs[0].locator["line_start"] for a in atoms]
    assert line_starts == sorted(line_starts)


def test_pretty_printed_nested_fixture(tmp_path: Path):
    text = json.dumps(FIXTURE, indent=2, ensure_ascii=False)
    atoms = _parse(tmp_path, text)
    _check(text, atoms)
    # Three identical "quantity": 1 values each point at their own line.
    qty = [a for a in atoms if a.raw_text.startswith("equipment.items[") and ".quantity:" in a.raw_text]
    assert len(qty) == 3
    q_lines = [a.source_refs[0].locator["line_start"] for a in qty]
    assert len(set(q_lines)) == 3
    for a in qty:
        assert text.split("\n")[a.source_refs[0].locator["line_start"] - 1].strip() == '"quantity": 1,'


def test_minified_and_ascii_escaped(tmp_path: Path):
    text = json.dumps(FIXTURE)  # one line, \u escapes
    atoms = _parse(tmp_path, text)
    _check(text, atoms)
    assert {a.source_refs[0].locator["line_start"] for a in atoms} == {1}


def test_crlf_bom_and_odd_spacing(tmp_path: Path):
    body = json.dumps(FIXTURE, indent=4, ensure_ascii=False).replace("\n", "\r\n").replace(": ", " :  ")
    atoms = _parse(tmp_path, "﻿" + body)
    # utf-8-sig strips the BOM, as a browser's decoder does.
    for a in atoms:
        loc = a.source_refs[0].locator
        frag = body[loc["char_start"]:loc["char_end"]]
        assert json.loads(frag) == _resolve(FIXTURE, loc["json_pointer"])
    starts = [a.source_refs[0].locator["char_start"] for a in atoms]
    assert starts == sorted(starts)


def test_jsonl_records(tmp_path: Path):
    text = '{"a": 1, "b": "x"}\n\nnot json\n{"a": 1, "b": "y"}\n'
    atoms = _parse(tmp_path, text, "events.jsonl")
    got = {a.source_refs[0].locator["json_pointer"]: a.source_refs[0].locator for a in atoms}
    assert got["/line1/a"]["line_start"] == 1
    assert got["/line2/b"]["line_start"] == 4
    assert text[got["/line2/b"]["char_start"]:got["/line2/b"]["char_end"]] == '"y"'


def test_atom_ids_do_not_move(tmp_path: Path):
    """Positions are added to the locator, not to the atom's identity."""
    text = json.dumps({"a": {"b": 1}}, indent=2)
    pretty = _parse(tmp_path, text)
    flat = _parse(tmp_path, json.dumps({"a": {"b": 1}}))
    assert [a.id for a in pretty] == [a.id for a in flat]
    assert pretty[0].source_refs[0].locator["line_start"] != flat[0].source_refs[0].locator["line_start"]


def test_envelope_reading_order_is_file_order(tmp_path: Path):
    from app.core.orbitbrief_envelope import _in_reading_order

    text = json.dumps(FIXTURE, indent=2, ensure_ascii=False)
    atoms = _parse(tmp_path, text)
    shuffled = list(reversed(atoms))
    ordered = _in_reading_order(shuffled, [{"artifact_id": atoms[0].artifact_id}])
    assert [a.id for a in ordered] == [a.id for a in atoms]
