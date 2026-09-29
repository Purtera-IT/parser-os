# -*- coding: utf-8 -*-
"""Every SHAPE stage that can drop an atom, watched during a real compile.

The stages only fire on the right material: `pasted_note_dedup` and
`quoted_history_dedup` need email, and COPPER_001 is PDF-heavy, so on that
deal they never ran at all. Auditing them means running a deal that has what
they act on.

Wraps each stage, records what went in and what came out, and for every atom
dropped reports what it held that no survivor holds. That is the only question
that matters here: a fold is fine, a deletion is not.

    DEAL=<uuid> python _shape_audit.py
"""
import os, re, sys, tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "shape_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "shape_o.db"))

DEAL = os.environ["DEAL"]
MAX = int(os.environ.get("MAX_ARTIFACTS", "40"))

CAP: dict[str, dict] = {}


def watch(module, name: str):
    """Record one stage's atoms in and out, without changing what it does."""
    real = getattr(module, name)

    def wrapped(*a, **kw):
        before = list(a[0]) if a and isinstance(a[0], list) else []
        out = real(*a, **kw)
        after = out[0] if isinstance(out, tuple) else out
        if isinstance(after, list) and before:
            CAP.setdefault(name, {"before": before, "after": list(after)})
        return out

    setattr(module, name, wrapped)
    return wrapped


def main() -> None:
    from azure.storage.blob import BlobServiceClient

    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    proj = Path(tempfile.mkdtemp(prefix="shape_")) / DEAL[:8]
    (proj / "artifacts").mkdir(parents=True)
    n = 0
    for b in list(cc.list_blobs(name_starts_with=f"deals/{DEAL}/artifacts/"))[:MAX]:
        p = proj / "artifacts" / b.name.split("/")[-1]
        try:
            p.write_bytes(cc.get_blob_client(b.name).download_blob().readall())
            n += 1
        except Exception:
            pass
    print("staged %d artifacts for %s" % (n, DEAL[:8]))

    import app.core.email_threading as ET
    import app.core.pasted_note_dedup as PN
    import app.core.entity_hygiene as EH
    import app.core.entity_resolution as ER
    import app.core.compiler as C

    for mod, fn in ((PN, "collapse_pasted_note_duplicates"),
                    (ET, "dedup_quoted_history"),
                    (EH, "drop_execution_boilerplate"),
                    (ER, "collapse_duplicate_atoms")):
        w = watch(mod, fn)
        if hasattr(C, fn):
            setattr(C, fn, w)

    from app.core.compiler import compile_project
    compile_project(project_dir=proj / "artifacts", project_id=DEAL[:8],
                    allow_errors=True, allow_unverified_receipts=True, use_cache=False)

    print("\n%-34s %7s %7s %8s" % ("stage", "in", "out", "dropped"))
    for name, cap in CAP.items():
        b, a = cap["before"], cap["after"]
        print("%-34s %7d %7d %8d" % (name, len(b), len(a), len(b) - len(a)))

    for name, cap in CAP.items():
        b, a = cap["before"], cap["after"]
        ai = {id(x) for x in a}
        dropped = [x for x in b if id(x) not in ai]
        if not dropped:
            continue
        # What did the dropped atoms hold that NOTHING surviving holds?
        # Whitespace-normalised, both sides. An earlier version compared exact
        # strings and reported a PM's document reference as lost because the
        # dropped copy had two spaces after a colon and the survivor had one.
        # A fold that only changes spacing has not deleted anything.
        norm = lambda t: " ".join((t or "").split())
        surviving_text = {norm(getattr(x, "raw_text", "")) for x in a}
        surviving_nums = set()
        for x in a:
            surviving_nums.update(re.findall(r"\d+(?:[.,]\d+)*", getattr(x, "raw_text", "") or ""))
        gone_text = 0
        gone_nums = Counter()
        for d in dropped:
            t = norm(getattr(d, "raw_text", ""))
            if t and t not in surviving_text:
                gone_text += 1
                for num in re.findall(r"\d+(?:[.,]\d+)*", t):
                    if num not in surviving_nums:
                        gone_nums[num] += 1
        print("\n%s: %d dropped, %d whose exact text survives nowhere" % (name, len(dropped), gone_text))
        if gone_nums:
            print("   figures that leave the compile entirely: %d  %s"
                  % (len(gone_nums), sorted(gone_nums, key=lambda k: -gone_nums[k])[:10]))
            for d in dropped[:3]:
                t = (getattr(d, "raw_text", "") or "").strip()
                if t and t not in surviving_text and re.search(r"\d", t):
                    print("     %s" % t[:96])
        else:
            print("   no figure leaves the compile")


if __name__ == "__main__":
    main()


def show_unique_drops(stage="dedup_quoted_history", limit=12):
    """The dropped atoms whose exact text survives nowhere -- are they content?"""
    cap = CAP.get(stage)
    if not cap:
        print("no capture for %s" % stage); return
    b, a = cap["before"], cap["after"]
    ai = {id(x) for x in a}
    norm = lambda t: " ".join((t or "").split())
    surviving = {norm(getattr(x, "raw_text", "")) for x in a}
    out = []
    for d in b:
        if id(d) in ai:
            continue
        t = norm(getattr(d, "raw_text", ""))
        if t and t not in surviving:
            out.append(t)
    print("\n%s: %d dropped whose text survives nowhere" % (stage, len(out)))
    for t in out[:limit]:
        print("   %s" % t[:110].replace("\n", " "))
