"""A note's lines pair with their own list in the mail, not the first copy.

One quoted message carries a request list and, a few lines below, a restated
list of it; several restated items repeat request items word for word. Two
earlier notes hold the two lists. Paired by the first copy of each line's
words, the restated note took the request's copies of the shared items, so
the restated list's heading (and the items only it has) folded onto its
note while its other items stayed: a heading folded away from its own list.
A heading and its list fold together onto the note that holds them.

Synthetic text only.
"""
from __future__ import annotations

from app.core.pasted_note_dedup import collapse_pasted_note_duplicates
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.suppression_ledger import take_folds

REQUEST = [
    "Upkeep of the office network, servers and storage for the client",
    "Support for wired and wireless network gear",
    "Support for file servers and storage arrays",
    "Help with the hypervisor cluster at the head office",
    "Desk-side support for staff laptops and phones",
    "Response to outages and service tickets",
]
RESTATED = [
    "Ongoing care of their network, server and storage estate",
    "Support for wired and wireless network gear",
    "Support for file servers and storage arrays",
    "Care of their virtual machine hosts and clusters",
    "Desk-side support for staff laptops and phones",
    "Response to outages and service tickets",
]


def _atom(text: str, artifact: str, n: int, value: dict, flags: list[str] | None = None, line: int | None = None):
    loc = {"line_start": line, "message_index": value.get("message_index")} if line is not None else {}
    return EvidenceAtom(
        id=f"atm_{artifact}_{n}", project_id="p", artifact_id=artifact,
        atom_type=AtomType.scope_item, raw_text=text, normalized_text=text.lower(), value=value,
        entity_keys=[],
        source_refs=[SourceRef(id=f"src_{artifact}_{n}", artifact_id=artifact, artifact_type=ArtifactType.txt,
                               filename=f"{artifact}.txt", locator=loc, extraction_method="test", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.7, review_flags=flags or [], review_status=ReviewStatus.auto_accepted, parser_version="test",
    )


def _note(artifact: str, lines: list[str], date: str):
    atoms = [_atom(t, artifact, i, {"kind": "hubspot_note_body"}, ["hubspot_note_parser"]) for i, t in enumerate(lines)]
    atoms.append(_atom(f"note_id={artifact} | date={date}", artifact, 99,
                       {"kind": "hubspot_note_meta", "date": date}, ["hubspot_note_parser"]))
    return atoms


def _mail():
    lines = REQUEST + ["Sites", "Springfield", "Shelbyville"] + RESTATED
    value = {"kind": "email_body_line", "message_index": 4, "quoted": True,
             "email_thread": {"date": "Mon, 21 Sep 2026 10:00:00 +0000"}}
    return [_atom(t, "art_mail", i, dict(value), line=40 + i) for i, t in enumerate(lines)]


def test_restated_heading_and_its_items_fold_onto_their_own_note():
    request_note = _note("art_note_req", REQUEST, "2026-05-29T18:58:14Z")
    restated_note = _note("art_note_rest", RESTATED, "2026-05-29T18:58:26Z")
    mail = _mail()
    take_folds()
    for order in ([*request_note, *restated_note], [*restated_note, *request_note]):
        take_folds()
        atoms = [a.model_copy(deep=True) for a in mail + order]
        kept, dropped = collapse_pasted_note_duplicates(atoms)
        folds = take_folds()
        by_line = {a.source_refs[0].locator["line_start"]: a for a in atoms if a.artifact_id == "art_mail"}
        request_lines = [by_line[40 + i] for i in range(len(REQUEST))]
        restated_lines = [by_line[40 + len(REQUEST) + 3 + i] for i in range(len(RESTATED))]
        # Every line of each list folds, onto the note that holds that list.
        for line in request_lines:
            assert id(line) in folds and folds[id(line)][1].artifact_id == "art_note_req", line.raw_text
        for line in restated_lines:
            assert id(line) in folds and folds[id(line)][1].artifact_id == "art_note_rest", line.raw_text
        # The site lines between them are no note's.
        assert all(by_line[40 + len(REQUEST) + k] in kept for k in range(3))
