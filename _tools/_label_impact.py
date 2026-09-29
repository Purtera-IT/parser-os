# -*- coding: utf-8 -*-
"""Do this deal's GOLD labels still attach after a parser change?

`label_key` is sha256(deal | filename | page | text). Three of those four are
location; the fourth is the atom's own words. So a label survives a rename or a
re-fetch, and stops resolving the moment the parser words the fact differently.
That makes re-attachment the honest measure of a parse change: an atom count
can lose one fact and gain another and look unchanged, but a gold label cannot
be argued with -- a person wrote it against text the parser used to produce.

Parses the deal's artifacts with whatever tree it is run from and reports how
many gold labels find a home. Run it on both trees and compare.

    DEAL=<uuid> python _label_impact.py <out.json>
"""
import hashlib, json, os, sys, tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(Path(os.environ.get("TREE", r"C:\Users\lilli\parser-os-labeling"))))

DEAL = os.environ["DEAL"]

from app.core.label_key import label_key  # noqa: E402


def gold_labels() -> list[tuple[str, str]]:
    import psycopg2
    dsn = (HERE / ".pgurl").read_text(encoding="utf-8").strip()
    with psycopg2.connect(dsn) as cx, cx.cursor() as cur:
        cur.execute(
            """
            SELECT label_key, COALESCE(labeler,'')
            FROM atom_labels
            WHERE deal_id = %s
              AND labeler LIKE '%%@%%'
              AND labeler NOT ILIKE '%%claude%%'
            """,
            (DEAL,),
        )
        return [(r[0], r[1]) for r in cur.fetchall()]


def parsed_keys() -> set[str]:
    from azure.storage.blob import BlobServiceClient
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    tmp = Path(tempfile.mkdtemp(prefix="li_"))
    paths = []
    for b in cc.list_blobs(name_starts_with=f"deals/{DEAL}/artifacts/"):
        p = tmp / b.name.split("/")[-1]
        p.write_bytes(cc.get_blob_client(b.name).download_blob().readall())
        paths.append(p)

    from app.parsers.registry import choose_parser
    keys: set[str] = set()
    for p in paths:
        try:
            parser, _m, _a = choose_parser(p)
            if parser is None:
                continue
            out = parser.parse_artifact_full(
                project_id=DEAL, artifact_id="a", path=p, domain_pack=None)
            for a in getattr(out, "atoms", out):
                text = getattr(a, "raw_text", None) or getattr(a, "text", "") or ""
                loc = getattr(a, "locator", None) or {}
                page = loc.get("page") if isinstance(loc, dict) else None
                keys.add(label_key(DEAL, p.name, page, text))
        except Exception as exc:
            print("  !! %s: %s" % (p.name[:50], type(exc).__name__), file=sys.stderr)
    return keys


def main() -> None:
    gold = gold_labels()
    keys = parsed_keys()
    resolved = [k for k, _ in gold if k in keys]
    out = {
        "deal": DEAL,
        "gold_total": len(gold),
        "resolved": len(resolved),
        "missing": sorted(k for k, _ in gold if k not in keys),
        "atom_keys": len(keys),
    }
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("gold labels: %d   resolved: %d   parser produced %d distinct keys"
          % (len(gold), len(resolved), len(keys)))


if __name__ == "__main__":
    main()
