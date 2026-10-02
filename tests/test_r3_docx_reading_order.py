"""R3 (010087 / 010003): .docx atoms must read in document order and sit in
the right section.

* Every docx atom carries a document-order locator (``block_index`` = body
  element position, ``line_start`` = reading position) so the envelope's
  ``_in_reading_order`` and the Platform-infra walk (which sorts on
  page|row, block_index, line_start, sentence_index) restore document order
  -- content controls, tables and paragraphs interleaved.
* Sibling headings (same style / same all-caps shape) replace each other on the
  heading stack instead of nesting, so PMO duties after "PURTERA
  RESPONSIBILITIES" are not typed as exclusions of an earlier OUT OF SCOPE.
* Heading atoms carry their own section_path (parent chain + the heading).
"""

from __future__ import annotations

from types import SimpleNamespace

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from app.core.orbitbrief_envelope import _in_reading_order
from app.parsers.docx_parser import DocxParser

W = nsdecls("w")

ORDER = [
    "Intro: the contractor shall survey the building before mobilizing crews.",
    "Step 01 the contractor shall install the core switch stack in the MDF.",
    "Step 02 the contractor shall provide 40 hours of onsite cabling labor.",
    "Step 03 Install fiber patch panel | in rack 3 near the MDF",
    "Step 04 Label every cable | at both ends with printed labels",
    "Step 05 the contractor shall terminate all drops on new patch panels.",
    "Step 06 the contractor shall mount twelve access points in the gym.",
    "Step 07 Certify each drop | with a Fluke tester and report",
    "Step 08 the contractor shall remove the abandoned coax runs.",
]


def _r(t: str) -> str:
    return f'<w:r><w:t xml:space="preserve">{t}</w:t></w:r>'


def _p(t: str) -> str:
    return f"<w:p {W}>{_r(t)}</w:p>"


def _tbl(rows: list[tuple[str, str]]) -> str:
    trs = "".join(
        f"<w:tr><w:tc><w:p>{_r(a)}</w:p></w:tc><w:tc><w:p>{_r(b)}</w:p></w:tc></w:tr>" for a, b in rows
    )
    return f"<w:tbl {W}><w:tblPr/><w:tblGrid><w:gridCol/><w:gridCol/></w:tblGrid>{trs}</w:tbl>"


def _sdt(inner: str) -> str:
    return f"<w:sdt {W}><w:sdtPr/><w:sdtContent>{inner}</w:sdtContent></w:sdt>"


def _build_interleaved(path) -> None:
    doc = Document()
    doc.add_paragraph(ORDER[0])
    sect = doc.element.body[-1]
    sect.addprevious(parse_xml(_sdt(_p(ORDER[1]) + _p(ORDER[2]))))
    a, b = ORDER[3].split(" | ")
    c, d = ORDER[4].split(" | ")
    sect.addprevious(parse_xml(_tbl([(a, b), (c, d)])))
    sect.addprevious(parse_xml(_p(ORDER[5])))
    sect.addprevious(parse_xml(_sdt(_p(ORDER[6]))))
    e, f = ORDER[7].split(" | ")
    sect.addprevious(parse_xml(_sdt(_tbl([(e, f)]))))
    sect.addprevious(parse_xml(_p(ORDER[8])))
    doc.save(path)


def _atoms(path):
    out = DocxParser().parse_artifact("p", "a", path)
    return out if isinstance(out, list) else out.atoms


def _rank(texts: list[str]) -> list[int]:
    ranks = []
    for t in texts:
        for i, o in enumerate(ORDER):
            key = o.split(" | ")[0][:20]
            if key in t:
                ranks.append(i)
                break
    return ranks


def _walk_sort(atoms):
    """Mirror of platform-infra atom-labeling.js row sort (page|row,
    block_index|row_index|row, line_start, sentence_index)."""

    def num(v, fb):
        if v is None:
            return 0  # JS Number(null) == 0
        try:
            return float(v)
        except (TypeError, ValueError):
            return fb

    def key(a):
        loc = a.source_refs[0].locator if a.source_refs else {}
        miss = object()
        g = lambda k: loc[k] if k in loc else miss  # noqa: E731

        def n(k, fb):
            v = g(k)
            return fb if v is miss else num(v, fb)

        return (
            n("page", n("row", 0)),
            n("block_index", n("row_index", n("row", 0))),
            n("line_start", 0),
            n("sentence_index", 0),
        )

    return sorted(atoms, key=key)


def test_atoms_emitted_in_document_order(tmp_path) -> None:
    path = tmp_path / "SOW.docx"
    _build_interleaved(path)
    atoms = _atoms(path)
    ranks = _rank([a.raw_text for a in atoms])
    assert sorted(set(ranks)) == list(range(len(ORDER))), ranks
    assert ranks == sorted(ranks), ranks


def test_every_atom_has_document_order_locator(tmp_path) -> None:
    path = tmp_path / "SOW.docx"
    _build_interleaved(path)
    for a in _atoms(path):
        loc = a.source_refs[0].locator
        assert isinstance(loc.get("block_index"), int), (a.raw_text, loc)
        assert isinstance(loc.get("line_start"), int), (a.raw_text, loc)


def test_envelope_reading_order_is_document_order(tmp_path) -> None:
    path = tmp_path / "SOW.docx"
    _build_interleaved(path)
    atoms = _atoms(path)
    # Scramble first: the envelope must restore order from the locator alone.
    scrambled = sorted(atoms, key=lambda a: a.id)
    ordered = _in_reading_order(scrambled, [{"artifact_id": "a"}])
    ranks = _rank([a.raw_text for a in ordered])
    assert ranks == sorted(ranks), ranks


def test_walk_sort_is_document_order(tmp_path) -> None:
    path = tmp_path / "SOW.docx"
    _build_interleaved(path)
    atoms = _atoms(path)
    ordered = _walk_sort(sorted(atoms, key=lambda a: a.id))
    ranks = _rank([a.raw_text for a in ordered])
    assert ranks == sorted(ranks), ranks
