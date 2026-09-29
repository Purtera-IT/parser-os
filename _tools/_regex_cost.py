# -*- coding: utf-8 -*-
"""Where regex time actually goes, per module and per call site.

"Costing us tons of time" has to be measured, not guessed: the phase map says
ENRICH is 32.5% of the compile, but that does not say whether the time is in
the regexes or in the work around them.

cProfile attributes time to the CALLER of each re method, so this reports the
modules and lines that spend the compile's time on pattern matching.

    DEAL=<uuid> python _regex_cost.py
"""
from __future__ import annotations

import cProfile
import os
import pstats
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "rc_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "rc_o.db"))


def main() -> None:
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    deal = os.environ["DEAL"]
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    proj = Path(tempfile.mkdtemp(prefix="rc_")) / "d"
    (proj / "artifacts").mkdir(parents=True)
    for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:25]:
        try:
            (proj / "artifacts" / b.name.split("/")[-1]).write_bytes(
                cc.get_blob_client(b.name).download_blob().readall())
        except Exception:
            pass

    pr = cProfile.Profile()
    pr.enable()
    C.compile_project(project_dir=proj / "artifacts", project_id=deal[:8],
                      allow_errors=True, allow_unverified_receipts=True,
                      use_cache=False)
    pr.disable()

    st = pstats.Stats(pr)
    total = st.total_tt
    re_calls = 0
    re_time = 0.0
    callers: Counter = Counter()
    times: Counter = Counter()
    for func, (cc_, nc, tt, ct, callers_map) in st.stats.items():
        fname, _line, name = func
        is_re = ("sre_compile" in fname or "/re/" in fname.replace("\\", "/")
                 or fname.endswith("re.py") or "re\_compiler" in fname
                 or name in ("search", "match", "findall", "finditer", "sub", "fullmatch", "split")
                 and ("re" in fname or "sre" in fname))
        if not is_re:
            continue
        re_calls += nc
        re_time += tt
        for (cf, cl, cn), (_pc, pnc, ptt, pct) in (callers_map or {}).items():
            if "parser-os" not in cf.replace("\\", "/"):
                continue
            key = "%s:%d %s" % (Path(cf).name, cl, cn)
            callers[key] += pnc
            times[key] += pct

    print("\ncompile total %.1fs; regex machinery %.2fs in %d calls (%.1f%%)"
          % (total, re_time, re_calls, 100 * re_time / total if total else 0))
    print("\n%-52s %10s %9s" % ("call site", "re calls", "sec"))
    for key, sec in times.most_common(20):
        print("%-52s %10d %9.3f" % (key[:52], callers[key], sec))

    mods: Counter = Counter()
    for key, sec in times.items():
        mods[key.split(":")[0]] += sec
    print("\n%-34s %9s" % ("module", "sec in regex"))
    for m, sec in mods.most_common(12):
        print("%-34s %9.3f" % (m, sec))


if __name__ == "__main__":
    main()
