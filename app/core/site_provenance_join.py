"""A per-site document tells you which site its rows are about.

Marion County's compile resolved all ten schools and then linked nothing to
them: of 486 atoms, exactly 10 carried a ``site:`` key, and those 10 WERE the
site atoms. Every access window, contract term and scope item sat unattached,
so the Deal Kit could say "there are ten schools" and "there is an 8am-3pm
access window" but never "Johnakin Middle opens at 8am".

The cause is that the only linker was a NAME MENTION matcher: an atom is joined
to a site when the site's name appears in the atom's own text. In a per-site
SOW that name appears once -- in the property block that becomes the site atom
-- and never again. The access-window row reads "Days of Operation | Monday -
Friday"; it does not repeat the school. So the matcher linked zero of them.

This joins on PROVENANCE instead. When a document resolves to exactly one
physical site, every atom from that document is about that site -- not by
similarity, but because that is what a per-site document IS. The association is
already carried by artifact_id; nothing needs to be inferred.

Embeddings would be the wrong instrument here and would be actively harmful:
these ten SOWs are one template, so the access-window rows are near-identical
across all of them. Any similarity measure links Johnakin's window to
Palmetto's with high confidence. Semantically identical rows belonging to
DIFFERENT sites are the norm in this corpus. Similarity answers "which text
looks alike", and the question is "which site is this about".

The guard that carries the weight is `exactly one`. A multi-site document -- a
locations list, a master spreadsheet -- propagates NOTHING, because smearing
ten schools across every row is worse than leaving them unlinked.
"""

from __future__ import annotations

from typing import Any

_SITE_PREFIX = "site:"


def _keys(atom: Any) -> list[str]:
    return [str(k) for k in (getattr(atom, "entity_keys", None) or [])]


def _site_keys(atom: Any) -> set[str]:
    return {k for k in _keys(atom) if k.startswith(_SITE_PREFIX)}


def _atom_type(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return str(getattr(at, "value", at) or "")


def document_site_map(atoms: list[Any]) -> dict[str, str]:
    """artifact_id -> the single ``site:`` key that document resolves to.

    Documents resolving to zero or to more than one site are absent, which is
    the whole safety property: only an unambiguous document may speak for its
    rows.
    """
    by_artifact: dict[str, set[str]] = {}
    for atom in atoms:
        if not _atom_type(atom).endswith("physical_site"):
            continue
        art = str(getattr(atom, "artifact_id", "") or "")
        if not art:
            continue
        by_artifact.setdefault(art, set()).update(_site_keys(atom))
    return {art: next(iter(keys)) for art, keys in by_artifact.items() if len(keys) == 1}


def join_atoms_to_document_site(atoms: list[Any]) -> int:
    """Give every atom of a single-site document that document's site key.

    Returns the number of atoms newly linked. Atoms that already carry a
    ``site:`` key are left alone -- a direct assertion outranks provenance.
    """
    site_of = document_site_map(atoms)
    if not site_of:
        return 0

    linked = 0
    for atom in atoms:
        art = str(getattr(atom, "artifact_id", "") or "")
        key = site_of.get(art)
        if not key:
            continue
        if _site_keys(atom):
            # Already attributed -- by its own text, or because it IS the site.
            continue
        try:
            atom.entity_keys = sorted(set(_keys(atom)) | {key})
        except Exception:
            continue
        # Stamp HOW this link was made, so an audit can tell "this row named
        # its school" apart from "this row lived in that school's file". A
        # join indistinguishable from a direct assertion is its own quiet
        # failure.
        try:
            prov = dict(getattr(atom, "decision_provenance", None) or {})
            prov.update({
                "source": "document_scope",
                "site_key": key,
                "rationale": (
                    "the document this atom came from resolves to exactly one "
                    "physical site"
                ),
            })
            atom.decision_provenance = prov
        except Exception:
            pass
        linked += 1
    return linked
