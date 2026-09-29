# -*- coding: utf-8 -*-
"""Who the signature reader finds, and whether they are distinct people.

Changing the cluster rules changes how many person records form. More records
is only better if they are more PEOPLE; the same person filed twice, or a
heading filed as a person, is a quality loss of the same kind as a deletion.

    DEAL=<uuid> python _people_check.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "pc_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "pc_o.db"))


def main() -> None:
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    deal = os.environ["DEAL"]
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    proj = Path(tempfile.mkdtemp(prefix="pc_")) / "d"
    (proj / "artifacts").mkdir(parents=True)
    for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:25]:
        try:
            (proj / "artifacts" / b.name.split("/")[-1]).write_bytes(
                cc.get_blob_client(b.name).download_blob().readall())
        except Exception:
            pass
    res = C.compile_project(project_dir=proj / "artifacts", project_id=deal[:8],
                            allow_errors=True, allow_unverified_receipts=True,
                            use_cache=False)
    atoms = list(res.atoms or [])
    print("total atoms: %d" % len(atoms))
    kinds = Counter(str(getattr(a, "atom_type", "")).split(".")[-1] for a in atoms)
    print("top atom types: %s" % kinds.most_common(8))

    people = [a for a in atoms
              if str(getattr(a, "atom_type", "")).endswith("stakeholder")]
    names = Counter()
    for a in people:
        v = getattr(a, "value", None) or {}
        names[str(v.get("name") or "").strip()] += 1
    print("\nstakeholder atoms: %d, distinct names: %d" % (len(people), len(names)))
    for n, c in names.most_common(30):
        print("   %-38s x%d" % (n[:38] or "<no name>", c))
    print("\nnames containing U+00A0: %s"
          % [n for n in names if "\u00a0" in n] or "none")


if __name__ == "__main__":
    main()
