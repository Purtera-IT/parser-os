# -*- coding: utf-8 -*-
"""For each phase-3 stage that drops atoms, WHICH figures left the compile.

`_phase3_audit.py` counts them; a count cannot tell a page footer from a bid
deadline. This prints the digits that no surviving atom states any more, with
the text they were read from, so each one can be read and judged.

    DEALS="uuid,uuid" python _figures_gone.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "fg_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "fg_o.db"))

norm = lambda t: " ".join((t or "").split()).lower()
figs = lambda t: set(re.findall(r"\d+(?:[.,]\d+)*", t or ""))


def _blob(atom) -> str:
    """An atom's words AND its structured value.

    Checking `raw_text` alone reports losses that never happened: a quoted
    message header is dropped as a duplicate while its timestamp lives on in
    the survivor's `value.email_thread`, and a site's address survives into
    `value` under different words. Both showed up as "a figure left the
    compile" until the value was read too.
    """
    import json as _j
    try:
        v = _j.dumps(getattr(atom, "value", None) or {}, default=str)
    except Exception:
        v = str(getattr(atom, "value", "") or "")
    return (getattr(atom, "raw_text", "") or "") + " " + v



def run(deal: str, max_artifacts: int = 25):
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    proj = Path(tempfile.mkdtemp(prefix="fg_")) / "d"
    (proj / "artifacts").mkdir(parents=True)
    for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:max_artifacts]:
        try:
            (proj / "artifacts" / b.name.split("/")[-1]).write_bytes(
                cc.get_blob_client(b.name).download_blob().readall())
        except Exception:
            pass
    return C.compile_project(project_dir=proj / "artifacts", project_id=deal[:8],
                             allow_errors=True, allow_unverified_receipts=True,
                             use_cache=False)


def main() -> None:
    total: Counter = Counter()
    for deal in [d.strip() for d in os.environ.get("DEALS", "").split(",") if d.strip()]:
        try:
            res = run(deal)
        except Exception as exc:
            print("  %s: %s: %s" % (deal[:8], type(exc).__name__, exc))
            continue

        kept_figs: set[str] = set()
        kept_text: set[str] = set()
        for x in (res.atoms or []):
            t = getattr(x, "raw_text", "")
            kept_figs |= figs(_blob(x))
            kept_text.add(norm(t))

        by_stage: dict[str, list] = defaultdict(list)
        for a in (getattr(res, "suppressed_atoms", None) or []):
            for f in (getattr(a, "review_flags", None) or []):
                if str(f).startswith("suppressed:"):
                    by_stage[str(f).split(":", 1)[1]].append(a)
                    break

        print("\n\n############ %s ############" % deal[:8])
        for stage, atoms in sorted(by_stage.items(), key=lambda kv: -len(kv[1])):
            rows = []
            for a in atoms:
                t = getattr(a, "raw_text", "")
                if norm(t) in kept_text:
                    continue
                lost = figs(t) - kept_figs
                if lost:
                    rows.append((sorted(lost), str(getattr(a, "atom_type", ""))[:28],
                                 " ".join(t.split())[:100]))
            if not rows:
                continue
            total[stage] += len(rows)
            print("\n=== %s: %d atom(s) took a figure with them ===" % (stage, len(rows)))
            for lost, at, t in rows[:25]:
                print("  %-22s %-26s %s" % (",".join(lost)[:22], at, t))
    print("\n\n=== totals ===")
    for k, n in total.most_common():
        print("  %-28s %4d" % (k, n))


if __name__ == "__main__":
    main()
