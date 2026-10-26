"""The main-heads nightly's report: queue -> rows -> refit -> held-out, one page.

    python -m ml.c3.main_heads_report --run ~/c3-local/main_heads_loop/runs/<day>

Reads what the run left behind (queue/summary.json, rows/index.json,
fit_<tag>.log, holdout.json) and writes report.json + report.md beside them.
Report-only: it names no checkpoint to serve; promoting a refit stays the
reviewed step (held-out here + full-line diff + golden 34/34).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_ROWS = re.compile(r"rows train=(\d+) validation=(\d+)")
_CKPT = re.compile(r"checkpoint (\S+) validation exact ([\d.]+) \(n=(\d+)\)")
_SAVED = re.compile(r"saved \S+: (\S+) validation exact ([\d.]+) -> ([\d.]+)")
_NOGAIN = re.compile(r"no epoch beat the checkpoint's ([\d.]+)")
_NONE = re.compile(r"no (\S+) labels in train or validation")


def parse_fit_log(text: str) -> dict[str, Any]:
    """What one slot_fit run said: rows, the checkpoint's validation exact and the refit's."""
    out: dict[str, Any] = {"status": "not run"}
    if m := _NONE.search(text):
        return {"status": "no labels in train or validation"}
    if m := _ROWS.search(text):
        out.update(train=int(m.group(1)), val=int(m.group(2)))
    if m := _CKPT.search(text):
        out.update(val_base=float(m.group(2)))
    if m := _SAVED.search(text):
        out.update(status="saved", val_base=float(m.group(2)), val_refit=float(m.group(3)))
    elif m := _NOGAIN.search(text):
        out.update(status="no gain on validation; nothing saved")
    elif "Traceback" in text:
        out["status"] = "failed"
    return out


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def build(run: Path, heads: list[str]) -> dict[str, Any]:
    fits = {}
    for h in heads:
        log = run / f"fit_{h.split(':', 1)[1]}.log"
        fits[h] = parse_fit_log(log.read_text()) if log.exists() else {"status": "not run"}
    return {"run": str(run), "queue": _load(run / "queue" / "summary.json"),
            "rows": _load(run / "rows" / "index.json"), "fits": fits,
            "holdout": _load(run / "holdout.json"), "final_checkpoint": (run / "checkpoint.txt").read_text().strip()
            if (run / "checkpoint.txt").exists() else None}


def _labels(rows: dict[str, Any] | None, key: str, part: str) -> str:
    s = ((rows or {}).get("heads") or {}).get(key, {}).get(part) or {}
    if not s:
        return "0"
    return f"{s.get('person', 0) + s.get('approved', 0)}" + (f" (+{s['draft']} draft)" if s.get("draft") else "")


def to_md(rep: dict[str, Any], heads: list[str]) -> str:
    q, rows, ho = rep.get("queue") or {}, rep.get("rows") or {}, rep.get("holdout") or {}
    L = [f"# Main heads nightly — {Path(rep['run']).name}", "",
         "Report-only. No checkpoint is served from this run.", "",
         "## 1 Queue (PM corrections → atom path)", ""]
    if q:
        L.append(f"{q.get('corrections')} corrections exported; {q.get('main_head_rows')} main-head rows, "
                 f"{q.get('trainable')} trainable over {q.get('deals')} deals (envelopes {q.get('envelopes')}).")
        L += ["", "| head | person | draft | skipped |", "|---|---|---|---|"]
        for k, c in (q.get("by_head") or {}).items():
            sk = ", ".join(f"{s[5:]} {n}" for s, n in sorted(c.items()) if s.startswith("skip:")) or "0"
            L.append(f"| {k} | {c.get('person', 0)} | {c.get('draft', 0)} | {sk} |")
        for k, why in (q.get("no_route") or {}).items():
            L.append(f"| {k} | - | - | {why} |")
    L += ["", "## 2 Rows (trained labels per split; drafts never train unless approved)", "",
          "| head | train | val | holdout |", "|---|---|---|---|"]
    for k in sorted((rows.get("heads") or {})):
        L.append(f"| {k} | {_labels(rows, k, 'train')} | {_labels(rows, k, 'val')} | {_labels(rows, k, 'holdout')} |")
    if rows.get("counts"):
        L.append("")
        L.append("Counts: " + ", ".join(f"{k} {v}" for k, v in sorted(rows["counts"].items())))
    L += ["", "## 3 Refit (one head at a time, epoch picked on val)", "",
          "| head | status | train rows | val rows | val base | val refit |", "|---|---|---|---|---|---|"]
    for h in heads:
        f = rep["fits"].get(h) or {}
        L.append(f"| {h} | {f.get('status')} | {f.get('train', '-')} | {f.get('val', '-')} | "
                 f"{f.get('val_base', '-')} | {f.get('val_refit', '-')} |")
    L += ["", "Pair heads (jdg:site, jdg:conflict -> rel:supports) are queued and written as rows but not "
          "refit here: slot_fit fits one-line heads only.", ""]
    L += ["", "## 4 Held-out (deals the refit never read)", ""]
    if not ho:
        L.append("Not run.")
    else:
        L += [f"{ho.get('files', 0)} held-out deal file(s).", "",
              "| head | n | majority | base exact | refit exact | verdict |", "|---|---|---|---|---|---|"]
        for k, v in (ho.get("heads") or {}).items():
            b, r = v.get("base") or {}, v.get("refit") or {}
            L.append(f"| {k} | {b.get('n', 0)} | {b.get('majority', '-')} | {b.get('exact', '-')} | "
                     f"{r.get('exact', '-')} | {v.get('verdict')} |")
    L += ["", f"Final refit (report-only): `{rep.get('final_checkpoint') or 'none'}`", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--heads", nargs="*", default=["col:label_type", "col:admission", "col:supplier",
                                                   "jdg:site_role", "jdg:facility", "read:address_level"])
    a = p.parse_args(argv)
    run = Path(a.run).expanduser()
    rep = build(run, a.heads)
    (run / "report.json").write_text(json.dumps(rep, indent=1))
    md = to_md(rep, a.heads)
    (run / "report.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
