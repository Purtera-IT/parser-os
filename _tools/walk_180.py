"""Pull 010180's freshest envelope and lay every atom out for labelling.

Keys are computed from the SAME (filename, page) that get written beside them.
On 010288, 31 of 130 labels do not recompute their own key because the drawing's
atoms were keyed under one filename and stored under another, so those labels
will not re-attach on a re-compile. The key is the one thing that is supposed to
survive a re-parse; it only does that if it is built from what is stored.

    python walk_180.py            # newest envelope in blob
    python walk_180.py --local    # the cached one, for offline work
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

from app.core.label_key import label_key

#: The deal being labelled. An env var, not a constant: these scripts were
#: written for 010180 and a hardcoded id here would silently write another
#: deal's labels under this deal's rows, which no audit would catch.
DEAL = os.environ.get("DEAL") or "c79db726-323e-41f8-899d-1d8ca29a579a"
HERE = Path(__file__).parent
OUT = HERE / os.environ.get("WALK", "walk_180.json")


def _from_blob() -> dict:
    from azure.storage.blob import BlobServiceClient

    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    client = BlobServiceClient.from_connection_string(conn)
    container = client.get_container_client("orbitbrief-artifacts")
    prefix = f"deals/{DEAL}/orbitbrief/"
    blobs = [b for b in container.list_blobs(name_starts_with=prefix)
             if b.name.endswith("envelope.json")]
    if not blobs:
        raise SystemExit(f"no envelope.json under {prefix}")
    newest = max(blobs, key=lambda b: b.last_modified)
    print(f"envelope: {newest.name}  ({newest.last_modified:%Y-%m-%d %H:%M})")
    return json.loads(container.get_blob_client(newest.name).download_blob().readall())



_RESULT_CACHE: dict = {}


def _parser_result() -> dict:
    """`parser-os/latest/result.json` -- the only place the suppression ledger
    lives; the orbitbrief envelope this walk is built from does not carry it."""
    if "r" in _RESULT_CACHE:
        return _RESULT_CACHE["r"]
    try:
        from azure.storage.blob import BlobServiceClient

        conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
        container = BlobServiceClient.from_connection_string(conn).get_container_client(
            "orbitbrief-artifacts")
        _RESULT_CACHE["r"] = json.loads(
            container.download_blob(f"deals/{DEAL}/parser-os/latest/result.json").readall())
    except Exception as exc:
        print(f"  (no suppression ledger: {type(exc).__name__})")
        _RESULT_CACHE["r"] = {}
    return _RESULT_CACHE["r"]


def _substance_gate_drops() -> list[dict]:
    """The substance gate's own removals, from the parser result.

    The orbitbrief envelope this walk is built from does not carry the
    suppression ledger; only `parser-os/latest/result.json` does.
    """
    try:
        from azure.storage.blob import BlobServiceClient

        result = _parser_result()
    except Exception as exc:                                   # offline, or no blob
        print(f"  (no suppression ledger: {type(exc).__name__})")
        return []
    out = [a for a in (result.get("suppressed_atoms") or [])
           if "suppressed:substance_gate" in (a.get("review_flags") or [])]
    seen: set[tuple] = set()
    unique = []
    for a in out:
        refs = a.get("source_refs") or []
        key = (" ".join(str(a.get("raw_text") or "").split()),
               (refs[0].get("filename") if refs else ""))
        if key in seen:
            continue                                            # same line, many replies
        seen.add(key)
        unique.append(a)
    print(f"  suppression ledger: {len(out)} gate drops, {len(unique)} distinct")
    return unique



#: The dedup stages, which collapse duplicates rather than judging them.
_DEDUP_STAGES = ("suppressed:quoted_history_dedup", "suppressed:semantic_dedup",
                 "suppressed:pasted_note_dedup", "suppressed:duplicate_atom_collapse",
                 "suppressed:pre_classify_dedup")


def _dedup_losers(result: dict, live_atoms: list[dict]) -> list[dict]:
    """The copies dedup folded away, one row per distinct text, each naming the
    atom it was folded into.

    Dedup is not the substance gate. It collapses duplicates into the richest
    copy and the winner keeps the losers' source_refs, so the FACT survives --
    which is why making it advisory would be wrong: it would mint 3,377 atoms
    on this deal for 179 distinct sentences and ask a labeller to judge "7 Penn
    building in NYC" thirty times.

    What was missing is not the atom, it is the ACCOUNT. Reading the workspace
    you could not tell whether a line you remembered was deleted or folded, and
    "7 Penn building in NYC" -- folded into the fuller address atom -- looked
    exactly like a line nobody had read. So each distinct loser comes through
    once, marked with the stage that took it and the live atom that now carries
    it, and a loser with no identifiable winner is flagged as such, because
    that is the case actually worth arguing about.
    """
    losers = [a for a in (result.get("suppressed_atoms") or [])
              if any(f in _DEDUP_STAGES for f in (a.get("review_flags") or []))]
    by_text: dict[str, dict] = {}
    for a in losers:
        text = " ".join(str(a.get("raw_text") or "").split())
        # Deliberately NOT the 24-char floor the admission head uses. That rule
        # is about whether a line nobody read is a fact; this list is about
        # accounting for something the parser DID read and then folded away,
        # and the first thing it hid at 24 was "7 Penn building in NYC" -- the
        # very atom that prompted the list. Two words and a letter is enough.
        if len(text) < 8 or len(text.split()) < 2:
            continue
        row = by_text.setdefault(text, {"atom": a, "copies": 0, "stages": set()})
        row["copies"] += 1
        row["stages"].update(f.split(":", 1)[1] for f in (a.get("review_flags") or [])
                             if f in _DEDUP_STAGES)

    live_by_text = {" ".join(str(x.get("text") or "").split()): x for x in live_atoms}
    live_keys = {}
    for x in live_atoms:
        for k in (x.get("entityKeys") or []):
            live_keys.setdefault(k, x)

    out = []
    for text, row in by_text.items():
        a = row["atom"]
        refs = a.get("source_refs") or []
        ref = refs[0] if refs else {}
        # Who carries this fact now? The same words if the survivor kept them,
        # otherwise an atom sharing its entity keys -- and if neither, say so.
        winner = live_by_text.get(text)
        if winner is None:
            for k in (a.get("entity_keys") or []):
                if k in live_keys:
                    winner = live_keys[k]
                    break
        out.append({
            "labelKey": label_key(DEAL, str(ref.get("filename") or ""),
                                  (ref.get("locator") or {}).get("page"), text),
            "atomId": a.get("atom_id") or a.get("id"),
            "artifactId": a.get("artifact_id"),
            "filename": str(ref.get("filename") or ""),
            "page": (ref.get("locator") or {}).get("page"),
            "text": text,
            "parserType": a.get("atom_type"),
            "section": [],
            "entityKeys": a.get("entity_keys") or [],
            "locator": ref.get("locator") or {},
            "suppressedBy": "+".join(sorted(row["stages"])),
            "copies": row["copies"],
            "mergedInto": (winner or {}).get("text"),
            "mergedIntoKey": (winner or {}).get("labelKey"),
        })
    orphans = sum(1 for r in out if not r["mergedInto"])
    print(f"  dedup losers: {len(losers)} atoms, {len(out)} distinct, "
          f"{orphans} with no identifiable survivor")
    return out


def main() -> None:
    if "--local" in sys.argv:
        env = json.loads((HERE / "envs" / f"{DEAL}.json").read_text(encoding="utf-8"))
        print("envelope: cached copy (STALE -- 2026-09-09)")
    else:
        env = _from_blob()

    docs = {d.get("artifact_id"): d for d in (env.get("documents") or [])}
    atoms = env.get("atoms") or []
    walk: list[dict] = []
    for a in atoms:
        doc = docs.get(a.get("artifact_id")) or {}
        filename = str(doc.get("filename") or "")
        locator = a.get("locator") or {}
        page = locator.get("page")
        text = " ".join(str(a.get("text") or "").split())
        walk.append({
            "labelKey": label_key(DEAL, filename, page, text),
            "atomId": a.get("id"),
            "artifactId": a.get("artifact_id"),
            "filename": filename,
            "page": page,
            "text": text,
            "parserType": a.get("atom_type"),
            "section": a.get("section_path") or [],
            "entityKeys": a.get("entity_keys") or [],
            "locator": locator,
        })

    # The gate's deletions are evidence nobody was shown.
    #
    # `apply_substance_gate` removes atoms it judges contextless and the
    # compiler files them in the parser result's retained-suppression ledger,
    # so nothing is destroyed -- but this walk read only `atoms`, so a labeller
    # never saw them and the judgement produced no training rows in either
    # direction. Live 010180: 26 atoms, 20 distinct texts, among them
    # "Definitely dude can help out." (the only statement in 42 documents about
    # what PurTera does) and "It's not letting me pull it up due to access
    # restraints." (a live blocker), next to "Sounds good." and a legal footer.
    #
    # That is a minute's sorting for a person and it is exactly the gold
    # `line_admission` has never had. Only the substance gate's own drops are
    # pulled in: the ledger also holds 3,450 rows of drawing wreckage, and
    # burying twenty arguable lines under those would defeat the point.
    for row in _dedup_losers(_parser_result(), walk):
        walk.append(row)

    suppressed = _substance_gate_drops()
    for a in suppressed:
        refs = a.get("source_refs") or []
        ref = refs[0] if refs else {}
        filename = str(ref.get("filename") or "")
        locator = ref.get("locator") or {}
        page = locator.get("page")
        text = " ".join(str(a.get("raw_text") or "").split())
        if not text:
            continue
        walk.append({
            "labelKey": label_key(DEAL, filename, page, text),
            "atomId": a.get("atom_id") or a.get("id"),
            "artifactId": a.get("artifact_id"),
            "filename": filename,
            "page": page,
            "text": text,
            "parserType": a.get("atom_type"),
            "section": [],
            "entityKeys": a.get("entity_keys") or [],
            "locator": locator,
            # The gate's verdict, carried so the labeller sees what it thought
            # and `line_admission` gets the old rule as a feature.
            "suppressedBy": "substance_gate",
        })

    OUT.write_text(json.dumps({
        "deal_id": DEAL,
        "compile_id": env.get("compile_id"),
        "generated_at": env.get("generated_at"),
        "documents": [{"artifactId": k, "filename": v.get("filename"),
                       "parser": v.get("parser_name"),
                       "authored_at": v.get("authored_at"),
                       "direction": v.get("direction"),
                       "outcome": (v.get("parse_outcome") or {}).get("status"),
                       "atom_count": sum(1 for w in walk if w["artifactId"] == k)}
                      for k, v in docs.items()],
        "atoms": walk,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    dupes = [k for k, n in collections.Counter(w["labelKey"] for w in walk).items() if n > 1]
    print(f"atoms: {len(walk)}  documents: {len(docs)}  duplicate keys: {len(dupes)}")
    print(f"types: {dict(collections.Counter(w['parserType'] for w in walk).most_common(12))}")
    print(f"wrote {OUT}")
    by_doc = collections.Counter(w["filename"] for w in walk)
    for name, n in by_doc.most_common():
        print(f"   {n:4d}  {name[:76]}")


if __name__ == "__main__":
    main()
