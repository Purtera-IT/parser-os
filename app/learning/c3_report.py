"""The C3 base's daily report, from what every compile recorded.

Each compile's ML manifest (``compile_capabilities.deflect_counts``) carries
what the C3 layer did: its mode and type list, whether it loaded, how long
the read took against the LLM, how many atoms it took, and in shadow, how
often it agreed with the compile per answer. The worker saves each deal's
latest compile as ``deals/<id>/parser-os/latest/result.json``. This pools
those into one report, written beside them as ``reports/c3/<date>.md`` (and
``.json``) and ``reports/c3/latest.md``:

* health: compiles read, loaded, over budget, short of memory, failed;
* speed: the base's read time against the LLM time, per compile and pooled;
* take: atoms the base typed, and the share of the atom-type stage it took;
* agreement (shadow): per answer, and the commonest disagreements;
* weak: answers whose agreement is under the bar, which is where to label.

Counts and times only; no line text is in the manifest or the report. Run by
the nightly job before its retrains (``nightly_retrain.main``), or by hand:
``python -m app.learning.c3_report``. Never raises into its caller.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

#: Agreement under this (shadow) puts an answer on the weak list.
WEAK_BELOW = float(os.environ.get("C3_REPORT_WEAK_BELOW", "0.85"))
#: Fewest shadow decisions an answer needs before it is judged.
MIN_N = int(os.environ.get("C3_REPORT_MIN_N", "20"))


def pool(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool the per-compile manifests' C3 and atom-type entries."""
    states: Counter = Counter()
    reasons: Counter = Counter()
    agree: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    pairs: Counter = Counter()
    modes: Counter = Counter()
    lists: Counter = Counter()
    read_ms, llm_ms, slow = [], [], 0
    taken = input_atoms = 0
    for m in manifests:
        dc = (m or {}).get("deflect_counts") or {}
        c3 = dc.get("c3")
        if not c3:
            continue
        typed = dc.get("typed_atom") or {}
        modes[c3.get("mode", "?")] += 1
        lists[",".join(c3.get("types") or []) or "(all value-light)"] += 1
        states[c3.get("state", "?")] += 1
        if c3.get("reason"):
            reasons[str(c3["reason"]).split(":")[0]] += 1
        if c3.get("read_ms") is not None:
            r = float(c3["read_ms"])
            read_ms.append(r)
            if typed.get("llm_ms") is not None:
                llm_ms.append(float(typed["llm_ms"]))
                slow += r > float(typed["llm_ms"])
        taken += int(typed.get("c3") or 0)
        input_atoms += int(c3.get("lines") or 0)
        shadow = c3.get("shadow") or {}
        for a, s in (shadow.get("by_answer") or {}).items():
            agree[a][0] += int(s.get("agree", 0))
            agree[a][1] += int(s.get("disagree", 0))
        pairs.update(shadow.get("disagree_pairs") or {})
    rates = {a: {"agree": g, "disagree": d, "n": g + d, "rate": round(g / max(1, g + d), 4)}
             for a, (g, d) in sorted(agree.items())}
    weak = sorted(a for a, v in rates.items() if v["n"] >= MIN_N and v["rate"] < WEAK_BELOW)
    thin = sorted(a for a, v in rates.items() if v["n"] < MIN_N)

    def _mid(xs: list[float]) -> float | None:
        return round(sorted(xs)[len(xs) // 2], 1) if xs else None

    return {
        "compiles": sum(states.values()), "modes": dict(modes), "type_lists": dict(lists),
        "health": dict(states), "failure_reasons": dict(reasons.most_common(5)),
        "speed": {"read_ms_median": _mid(read_ms), "llm_ms_median": _mid(llm_ms),
                  "read_ms_total": round(sum(read_ms), 1), "llm_ms_total": round(sum(llm_ms), 1),
                  "compiles_slower_than_llm": slow},
        "take": {"atoms_typed_by_base": taken, "atoms_read": input_atoms},
        "agreement": rates, "top_disagreements": dict(pairs.most_common(15)),
        "weak": weak, "thin": thin, "weak_below": WEAK_BELOW, "min_n": MIN_N,
    }


def markdown(rep: dict[str, Any], day: str) -> str:
    h, s, t = rep["health"], rep["speed"], rep["take"]
    ok = h.get("read", 0)
    lines = [f"# C3 base report, {day}", "",
             f"{rep['compiles']} compiles with the layer on or in shadow "
             f"({', '.join(f'{k} {v}' for k, v in rep['modes'].items()) or 'none'}).", "",
             "## Health", "",
             "| state | compiles |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(h.items())]
    if rep["failure_reasons"]:
        lines += ["", "Why it did not read: " + ", ".join(f"{k} {v}" for k, v in rep["failure_reasons"].items())]
    lines += ["", "## Speed", "",
              f"Median read {_s(s['read_ms_median'])} against median LLM time {_s(s['llm_ms_median'])}. "
              f"{s['compiles_slower_than_llm']} of {ok} reads took longer than the LLM.", "",
              "## What it took", "",
              f"{t['atoms_typed_by_base']} atoms typed by the base, from {t['atoms_read']} read.", ""]
    if rep["agreement"]:
        lines += ["## Agreement with the compile (shadow)", "",
                  "| answer | agree | disagree | rate |", "|---|---|---|---|"]
        lines += [f"| {a} | {v['agree']} | {v['disagree']} | {v['rate']:.3f} |"
                  for a, v in rep["agreement"].items()]
        lines += ["", "Commonest disagreements (base answer -> compile type):", ""]
        lines += [f"- {k}: {n}" for k, n in rep["top_disagreements"].items()]
        lines += ["", "## Where to label", ""]
        lines += [f"- **{a}**: agreement under {rep['weak_below']:.2f}. Label its disagreements."
                  for a in rep["weak"]] or ["- Nothing under the bar."]
        if rep["thin"]:
            lines += [f"- Too few decisions to judge (under {rep['min_n']}): {', '.join(rep['thin'])}."]
    return "\n".join(lines) + "\n"


def _s(ms: float | None) -> str:
    return "n/a" if ms is None else f"{ms / 1000:.1f}s"


def run(limit: int = 500) -> dict[str, Any] | None:
    """Read the saved compiles from blob, write the report beside them."""
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if not conn:
        print("[c3_report] no storage connection — skip")
        return None
    try:
        from azure.storage.blob import ContainerClient

        cc = ContainerClient.from_connection_string(
            conn, os.environ.get("AZURE_STORAGE_BLOB_CONTAINER", "orbitbrief-artifacts"))
        names = [b.name for b in cc.list_blobs(name_starts_with="deals/")
                 if b.name.endswith("/parser-os/latest/result.json")][:limit]
        manifests = []
        for nm in names:
            try:
                doc = json.loads(cc.download_blob(nm).readall())
                manifests.append(doc.get("compile_capabilities") or {})
            except Exception:
                continue
        rep = pool(manifests)
        if not rep["compiles"]:
            print(f"[c3_report] no compile in {len(names)} ran the C3 layer — skip")
            return rep
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        md = markdown(rep, day)
        for name, body in ((f"reports/c3/{day}.md", md), ("reports/c3/latest.md", md),
                           (f"reports/c3/{day}.json", json.dumps(rep, indent=1, sort_keys=True))):
            cc.upload_blob(name, body.encode("utf-8"), overwrite=True)
        print(f"[c3_report] {rep['compiles']} compiles -> reports/c3/{day}.md")
        return rep
    except Exception as e:
        print(f"[c3_report] skipped: {e}")
        return None


if __name__ == "__main__":
    run()
