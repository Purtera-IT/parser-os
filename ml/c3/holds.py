"""Explicit training-only holds. A held source/relation is unknown, never no-link.

Rows on disk remain intact. Loader copies remove held positive edges, their
notes and explicitly dependent cards; featurize also masks the no-link lesson.
No model input or unrelated line/site target is removed.
"""
from __future__ import annotations

import copy
from typing import Any

VERSION = 1


def mask_training_blob(blob: dict[str, Any], atoms: list[dict[str, Any]]
                       ) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    contract = blob.get("supervision_holds")
    if contract is None:
        return blob, []
    if not isinstance(contract, dict) or contract.get("version") != VERSION:
        raise ValueError("Unsupported supervision_holds version")
    groups = contract.get("groups")
    if not isinstance(groups, list):
        raise ValueError("supervision_holds groups must be a list")
    from .data import _blob_links

    result = copy.deepcopy(blob)
    rows = [dict(a, key=a.get("key") or a.get("label_key")) for a in atoms]
    atom_keys = {a["key"] for a in rows}
    held: set[tuple[str, str]] = set()
    source_relations: set[tuple[str, str]] = set()
    link_ids: set[str] = set()
    card_keys: set[str] = set()
    note_keys: set[str] = set()
    # Resolution is structural, independent of whether a file is approved.
    cards = copy.deepcopy(blob.get("judgments") or [])
    for j in cards:
        j["labeler"] = "hold-resolution"
    for group in groups:
        if (not isinstance(group, dict) or not group.get("group")
                or not isinstance(group.get("source_keys"), list)
                or not group["source_keys"] or not isinstance(group.get("relation"), str)
                or not group["relation"]):
            raise ValueError("Incomplete supervision hold group")
        rel = group["relation"]
        sources = set(group["source_keys"])
        if not all(isinstance(k, str) and k for k in sources):
            raise ValueError("Invalid held source key")
        ids = set(group.get("link_ids") or [])
        link_ids |= ids
        source_relations.update((key, rel) for key in sources)
        held.update((key, rel) for key in sources & atom_keys)
        note_keys.update(group.get("note_label_keys") or [])
        card_keys.update(group.get("judgment_targets") or [])
        for k in blob.get("links") or []:
            if (k.get("id") not in ids and k.get("supersedes") not in ids
                    and (k.get("from_key"), k.get("relation")) not in {(key, rel) for key in sources}):
                continue
            probe = dict(k, labeler="hold-resolution", relation=rel)
            edges, _ = _blob_links([probe], rows, judgments=cards)
            held.update((e["src"], rel) for e in edges)
    # Remove every saved/draft row on the held source/relation, not only a
    # particular positive (all target choices at that source are unknown).
    def keep_link(k):
        if k.get("id") in link_ids or k.get("supersedes") in link_ids:
            return False
        if (k.get("from_key"), k.get("relation")) in source_relations:
            return False
        # Other aliases resolving to the same line are suppressed after the
        # loader resolves all edges once, in featurize. Do not re-resolve an
        # entire deal once per edge here.
        return True

    result["links"] = [k for k in result.get("links") or [] if keep_link(k)]
    result["judgments"] = [j for j in result.get("judgments") or []
                            if j.get("target_key") not in card_keys]
    for lb in result.get("labels") or []:
        key = lb.get("label_key")
        lb["links"] = [k for k in lb.get("links") or []
                       if (key, k.get("relation")) not in held]
        if key in note_keys:
            lb.pop("note", None)
            lb["why_weight"] = 0.0
    result["held_relations"] = sorted(held)
    return result, sorted(held)
