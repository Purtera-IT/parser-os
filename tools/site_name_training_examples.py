"""Extract site-name training examples from a finished kit + its deal's
envelope (PUR-50). LOCAL files only; do not point this at real data without
the approvals real data needs. Nothing is trained here.

Ground truth is the kit's site name, keyed on the DOCUMENT the name is read
from: for each kit site we find the atoms whose own text contains the name
(dress-blind, so "St. Louis Office" and "ST LOUIS OFFICE" are one key) and
emit ``(document id, page, source span, kit site name, normalized key)``.
The model learns the NAME, not its phrasing: spelling variants share a
``normalized_key``.

Document scoping reuses the provenance join in
``app.core.site_provenance_join.document_site_map`` (a document that resolves
to exactly one site speaks for that site) rather than embeddings: per-site
documents are one template, so similarity would link the wrong site. When a
kit site carries a ``site_key``, only documents the join maps to that key, or
documents the join cannot scope (multi-site), are searched.

A kit site no document names yields an ABSTAIN example (``document_id`` None,
``abstain_reason`` "no_document_names_site"): the head must learn to abstain.

Input shapes
------------
kit JSON:      {"deal_id": "...", "sites": [{"name": "...", "site_key": "site:..."?}, ...]}
               (a bare list of names is also accepted)
envelope JSON: {"atoms": [{"id", "artifact_id", "atom_type", "text",
                "entity_keys", "locator": {"page": ...}}, ...]}

Usage::

    python tools/site_name_training_examples.py --kit KIT.json --envelope ENV.json [--out examples.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from app.core.site_provenance_join import document_site_map

ABSTAIN_NO_DOCUMENT = "no_document_names_site"


def normalized_key(name: Any) -> str:
    """Spelling-blind key: case, punctuation, separators, 'saint'/'st'."""
    s = str(name or "").lower().strip()
    s = re.sub(r"[\-_/.]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\bsaint\b", "st", s)
    return s


def _span_pattern(name: str) -> re.Pattern[str] | None:
    toks = normalized_key(name).split()
    if not toks:
        return None
    parts = [r"(?:st|saint)\.?" if t == "st" else re.escape(t) for t in toks]
    return re.compile(r"(?<![a-z0-9])" + r"[\s\-_/.,]+".join(parts) + r"(?![a-z0-9])", re.I)


def _kit_sites(kit: Any) -> list[dict[str, Any]]:
    sites = kit.get("sites") if isinstance(kit, dict) else kit
    out = []
    for s in sites or []:
        if isinstance(s, str):
            out.append({"name": s})
        elif isinstance(s, dict) and (s.get("name") or s.get("site_name")):
            out.append({"name": s.get("name") or s.get("site_name"), "site_key": s.get("site_key")})
    return out


def extract_examples(kit: Any, envelope: dict[str, Any]) -> list[dict[str, Any]]:
    atoms = [a for a in envelope.get("atoms") or [] if isinstance(a, dict)]
    shim = [
        SimpleNamespace(
            atom_type=a.get("atom_type"),
            artifact_id=a.get("artifact_id"),
            entity_keys=a.get("entity_keys") or [],
        )
        for a in atoms
    ]
    doc_site = document_site_map(shim)
    deal = (kit.get("deal_id") if isinstance(kit, dict) else None) or envelope.get("deal_id")
    examples: list[dict[str, Any]] = []
    for site in _kit_sites(kit):
        name = str(site["name"]).strip()
        key = normalized_key(name)
        pat = _span_pattern(name)
        want = site.get("site_key")
        found = False
        seen: set[tuple[Any, Any, int]] = set()
        for a in atoms:
            doc = a.get("artifact_id")
            if want and doc in doc_site and doc_site[doc] != want:
                continue  # provenance says this document is about another site
            text = str(a.get("text") or "")
            m = pat.search(text) if pat else None
            if not m:
                continue
            loc = a.get("locator") if isinstance(a.get("locator"), dict) else {}
            ident = (doc, loc.get("page"), m.start())
            if ident in seen:
                continue
            seen.add(ident)
            found = True
            examples.append({
                "deal_id": deal,
                "document_id": doc,
                "atom_id": a.get("id"),
                "page": loc.get("page"),
                "span": [m.start(), m.end()],
                "source_text": m.group(0),
                "kit_site_name": name,
                "normalized_key": key,
                "site_key": doc_site.get(doc) or want,
                "abstain_reason": None,
            })
        if not found:
            examples.append({
                "deal_id": deal, "document_id": None, "atom_id": None, "page": None,
                "span": None, "source_text": None, "kit_site_name": name,
                "normalized_key": key, "site_key": want, "abstain_reason": ABSTAIN_NO_DOCUMENT,
            })
    return examples


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--envelope", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    kit = json.loads(a.kit.read_text(encoding="utf-8"))
    env = json.loads(a.envelope.read_text(encoding="utf-8"))
    rows = extract_examples(kit, env)
    body = "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else "")
    if a.out:
        a.out.write_text(body, encoding="utf-8")
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
