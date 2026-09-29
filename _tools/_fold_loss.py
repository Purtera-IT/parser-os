# -*- coding: utf-8 -*-
"""When a twin is folded, does the survivor keep what the twin held?

`collapse_duplicate_atoms` keeps the higher-confidence copy and drops the
other. It does not MERGE. So if the dropped twin carried an entity key, a
receipt, a locator field or a value the survivor lacks, that fact leaves the
compile silently -- the atom count falls by a number nobody reads and the
content goes with it.

This runs the real stage over real atoms and reports, per fold, exactly what
the dropped twin held that the survivor does not.

    DEAL=<uuid> python _fold_loss.py        # or no DEAL for the COPPER fixture
"""
import os, sys, tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")


def load_atoms():
    from app.parsers.registry import choose_parser
    from app.core.ids import stable_id
    deal = os.environ.get("DEAL")
    if deal:
        from azure.storage.blob import BlobServiceClient
        conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
        cc = BlobServiceClient.from_connection_string(conn).get_container_client("orbitbrief-artifacts")
        tmp = Path(tempfile.mkdtemp()); paths = []
        for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:25]:
            p = tmp / b.name.split("/")[-1]
            p.write_bytes(cc.get_blob_client(b.name).download_blob().readall()); paths.append(p)
    else:
        root = Path(r"c:\Users\lilli\Downloads\purtera_copper_low_voltage_public_validation_packs"
                    r"\purtera_copper_low_voltage_validation_packs\real_data_cases"
                    r"\COPPER_001_SPRING_LAKE_AUDITORIUM\artifacts")
        paths = [p for p in sorted(root.rglob("*")) if p.is_file()]
    atoms = []
    for p in paths:
        try:
            parser, _m, _a = choose_parser(p)
            if parser is None:
                continue
            out = parser.parse_artifact_full(
                project_id="fold", artifact_id=stable_id("art", "fold", p.name),
                path=p, domain_pack=None)
            atoms.extend(getattr(out, "atoms", out))
        except Exception:
            pass
    return atoms


def facts(a) -> dict:
    """The things an atom carries that a compile would miss if they vanished."""
    out = {}
    for name in ("entity_keys", "receipts", "review_flags", "hints"):
        v = getattr(a, name, None)
        if v:
            out[name] = set(map(str, v)) if isinstance(v, (list, tuple, set)) else {str(v)}
    loc = getattr(a, "locator", None)
    if isinstance(loc, dict):
        out["locator"] = {f"{k}={loc[k]}" for k in loc if loc[k] not in (None, "", [], {})}
    val = getattr(a, "value", None)
    if isinstance(val, dict):
        out["value"] = {f"{k}={val[k]}" for k in val if val[k] not in (None, "", [], {})}
    sp = getattr(a, "section_path", None)
    if sp:
        out["section_path"] = {" > ".join(map(str, sp))}
    return out


def main() -> None:
    from app.core.entity_resolution import collapse_duplicate_atoms
    atoms = load_atoms()
    before = {id(a): a for a in atoms}
    kept = collapse_duplicate_atoms(list(atoms))
    kept_ids = {id(a) for a in kept}
    dropped = [a for a in atoms if id(a) not in kept_ids]
    print("atoms %d -> %d   dropped %d" % (len(atoms), len(kept), len(dropped)))
    if not dropped:
        return

    # Pair each dropped atom with the survivor that replaced it: same
    # artifact, same normalised text.
    def key(a):
        n = (getattr(a, "normalized_text", None) or getattr(a, "raw_text", "") or "").strip().lower()
        return (str(getattr(a, "artifact_id", "")), n)
    survivors = {}
    for a in kept:
        survivors.setdefault(key(a), a)

    lost = Counter()
    examples = []
    unpaired = 0
    for d in dropped:
        s = survivors.get(key(d))
        if s is None:
            unpaired += 1
            continue
        df, sf = facts(d), facts(s)
        for field, vals in df.items():
            missing = vals - sf.get(field, set())
            if missing:
                lost[field] += len(missing)
                if len(examples) < 8:
                    examples.append((field, sorted(missing)[:2],
                                     (getattr(d, "raw_text", "") or "")[:58]))
    print("dropped twins whose survivor could not be identified: %d" % unpaired)
    print("\nWHAT THE FOLD THREW AWAY (field -> count of values the survivor lacks):")
    if not lost:
        print("   nothing -- every dropped twin was covered by its survivor")
    for f, n in lost.most_common():
        print("   %-14s %d" % (f, n))
    for field, vals, text in examples:
        print("\n   %s lost %s" % (field, vals))
        print("     from: %s" % text)


if __name__ == "__main__":
    main()
