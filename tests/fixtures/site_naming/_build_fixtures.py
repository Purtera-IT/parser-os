"""Rebuild the site-naming fixtures from real envelopes on blob.

Nothing in this directory is authored. Every fixture is a trimmed copy of
an ``envelope.json`` the worker actually wrote, downloaded from
``purpulsedevstg01/orbitbrief-artifacts``. Hand-written site fixtures are
how four consecutive fixes for this area passed their tests and changed
nothing in production — the invented shape matched the code instead of
the artifact.

To refresh (envelopes as of 2026-08-09)::

    for D in 17dd11ae-27b7-4612-ae78-86667df90ecb \\
             0cc36784-1b81-4b8f-88a9-506a3a4bfab3 \\
             03783c65-be88-4fbb-bda2-95e9ee276a56 \\
             6bc2dc0a-ceaf-475d-940e-25bf7fac5531 ; do
      MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 az storage blob download \\
        --account-name purpulsedevstg01 -c orbitbrief-artifacts \\
        -n "deals/$D/orbitbrief/latest/envelope.json" -f "env_$D.json" \\
        --auth-mode login --no-progress -o none
    done
    python _build_fixtures.py

The trim keeps only what ``recover_site_display_names`` reads, at its real
value. Three reductions, each chosen because it cannot change the result:

  * site rows keep ``site`` / ``facility_name`` / ``aliases``;
  * exact-duplicate atom texts are dropped (the recovery sees a set of
    distinct strings — 0cc36784 repeats one sentence twelve times);
  * Clayton (6bc2dc0a) keeps all 437 real rows but only 60 atoms and no
    text, because every one of its rows is roster-anchored, so the
    recovery returns before it reads an atom. It is the regression guard:
    437 named sites in, 437 out, untouched.
"""
import json
import os

IDENTITY_FIELDS = ("name", "facility_name", "street_address", "address",
                   "names", "aliases", "alternative_names")
ROW_FIELDS = ("site", "facility_name", "aliases")

OUT = os.path.dirname(os.path.abspath(__file__))


def trim(env, keep_documents=True, keep_text=True, atom_cap=0):
    rows = (env.get("site_readiness") or {}).get("sites") or []
    out = {"sites": [{k: r[k] for k in ROW_FIELDS if k in r} for r in rows]}

    atoms = []
    seen_text = set()
    for a in env.get("atoms") or []:
        st = a.get("structured")
        identity = {k: st[k] for k in IDENTITY_FIELDS if k in st} if isinstance(st, dict) else {}
        text = (a.get("text") or "") if keep_text else ""
        if text in seen_text:
            text = ""
        elif text:
            seen_text.add(text)
        if not text and not identity:
            continue
        row = {}
        if text:
            row["text"] = text
        if identity:
            row["structured"] = identity
        atoms.append(row)
    out["atoms"] = atoms[:atom_cap] if atom_cap else atoms

    out["documents"] = ([{"structured": d.get("structured")}
                         for d in (env.get("documents") or [])]
                        if keep_documents else [])
    return out


#      fixture     source envelope                                  docs   text  cap
SPECS = [
    ("17dd11ae", "env_17dd11ae-27b7-4612-ae78-86667df90ecb.json", True,  True,  0),
    ("0cc36784", "env_0cc36784-1b81-4b8f-88a9-506a3a4bfab3.json", False, True,  0),
    ("03783c65", "env_03783c65-be88-4fbb-bda2-95e9ee276a56.json", False, True,  0),
    ("6bc2dc0a", "env_6bc2dc0a-ceaf-475d-940e-25bf7fac5531.json", False, False, 60),
]

if __name__ == "__main__":
    for name, path, keep_docs, keep_text, cap in SPECS:
        with open(path, encoding="utf-8") as fh:
            env = json.load(fh)
        trimmed = trim(env, keep_docs, keep_text, cap)
        dest = os.path.join(OUT, f"{name}.json")
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(trimmed, fh, ensure_ascii=False, separators=(",", ":"))
        print(f"{name}: sites={len(trimmed['sites'])} atoms={len(trimmed['atoms'])} "
              f"docs={len(trimmed['documents'])} size={os.path.getsize(dest) / 1024:.0f}KB")
