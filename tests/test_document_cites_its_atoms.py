"""A document's atoms are the ones that CITE it, not only the ones minted
while reading it.

`semantic_dedup` collapses duplicates across documents and the winner keeps
every loser's `source_ref`. The envelope then grouped atoms by the scalar
`atom.artifact_id`, so a collapsed atom appeared under the winner's document
and vanished from the others -- the atom still named those files, and they no
longer named it.

`_originating_sender` is the consumer that makes this visible, and it already
documents the contract: it scopes its walk to "THIS document's own refs"
BECAUSE dedup merges across documents. Grouping by the scalar id is what stopped
those refs ever arriving, so the defence had nothing to defend.

Live 010180, 42 documents, before any of today's changes: 9 atoms already span
several artifacts (a signature block is one fact quoted in 33 replies), and
grouping ref-aware moved `originated_by` on 32 of them from the CDW reseller who
forwarded the chain to the FlexTrade customer who started it. That direction is
the whole point of the function: attributing a customer's own material to an
intermediary is the deal-010215 bug it was written for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.orbitbrief_envelope import _originating_sender


@dataclass
class _Ref:
    artifact_id: str
    locator: dict
    id: str = "src"


@dataclass
class _Atom:
    artifact_id: str
    source_refs: list = field(default_factory=list)
    raw_text: str = ""
    value: dict = field(default_factory=dict)


def _group(atoms: list[_Atom], *, ref_aware: bool) -> dict[str, list[_Atom]]:
    out: dict[str, list[_Atom]] = {}
    for a in atoms:
        ids = {a.artifact_id}
        if ref_aware:
            ids |= {r.artifact_id for r in a.source_refs if r.artifact_id}
        for i in ids:
            out.setdefault(i, []).append(a)
    return out


#: One collapsed atom: minted in art_1, quoted in art_2. The customer sits
#: deeper in art_2's chain (index 4) than the reseller (index 1).
COLLAPSED = _Atom(
    artifact_id="art_1",
    raw_text="LS Srinivas | SVP & Global Head, Human Resources, FlexTrade Systems",
    source_refs=[
        _Ref("art_1", {"sender": "LS Srinivas <ls.srinivas@flextrade.com>", "message_index": 2}),
        _Ref("art_2", {"sender": "LS Srinivas <ls.srinivas@flextrade.com>", "message_index": 4}),
    ],
)
#: art_2's own shallow atom: the reseller who forwarded it.
FORWARDER = _Atom(
    artifact_id="art_2",
    raw_text="Erick Villalobos | Account Manager | CDW",
    source_refs=[_Ref("art_2", {"sender": "Erick Villalobos", "message_index": 1})],
)


def test_a_collapsed_atom_still_reaches_the_document_it_cites():
    groups = _group([COLLAPSED, FORWARDER], ref_aware=True)
    assert COLLAPSED in groups["art_2"]
    assert COLLAPSED in groups["art_1"]


def test_grouping_by_the_scalar_id_alone_loses_it():
    """The bug, stated as a test. Kept so the regression is legible if the
    grouping is ever narrowed back."""
    groups = _group([COLLAPSED, FORWARDER], ref_aware=False)
    assert COLLAPSED not in groups["art_2"]


def test_the_customer_who_started_the_chain_wins_not_the_reseller():
    """The consequence that matters: `originated_by` on art_2. With the atom
    absent, the deepest sender art_2 can see is the forwarder."""
    lost = _group([COLLAPSED, FORWARDER], ref_aware=False)
    assert _originating_sender(lost["art_2"], artifact_id="art_2") == "Erick Villalobos"

    kept = _group([COLLAPSED, FORWARDER], ref_aware=True)
    assert _originating_sender(kept["art_2"], artifact_id="art_2") == \
        "LS Srinivas <ls.srinivas@flextrade.com>"


def test_another_documents_indices_are_still_not_consulted():
    """Ref-aware grouping must not undo the scoping it depends on. art_3 quotes
    a different chain, and its index 9 must not outrank art_2's own index 4."""
    other = _Atom(
        artifact_id="art_3",
        raw_text="Somebody Else",
        source_refs=[_Ref("art_3", {"sender": "Somebody Else", "message_index": 9})],
    )
    groups = _group([COLLAPSED, FORWARDER, other], ref_aware=True)
    assert _originating_sender(groups["art_2"], artifact_id="art_2") == \
        "LS Srinivas <ls.srinivas@flextrade.com>"
