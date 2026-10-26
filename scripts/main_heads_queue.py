"""The corrections queue for the C3 super heads' MAIN heads (not the SOW heads).

    python scripts/main_heads_queue.py --corrections ~/c3-local/main_heads_loop/export/<day>/corrections.jsonl \\
        --envelopes ~/c3-local/main_heads_loop/export/<day>/envelopes --out ~/c3-local/main_heads_loop/runs/<day>/queue

Where the corrections live. ``POST /projects/{id}/feedback/correction`` runs
``app.core.pm_feedback.apply_pm_correction``: the correction goes into the
service's own per-process SQLite FeedbackStore (unreachable, lost on recycle)
and is mirrored to dev blob ``orbitbrief-artifacts/_feedback/corrections/<id>.json``
(``app.core.feedback_blob``; the gold TrainingRows go to ``_feedback/training_rows/``).
The blob mirror is the only durable copy, so this reads an export of it:
``scripts/main_heads_export.sh <dir>`` downloads it (read-only) and writes
``corrections.jsonl`` (one Correction per line) plus the latest envelope of
every deal a main-head correction names (``envelopes/<deal uuid>.json``).

Each exemplar of each correction becomes one queue row with:

* the super head it trains (``MAIN_HEADS``) and the PM's verdict;
* who made it: a person (an email, ``pm``, a user id) or a draft (an agent, a
  Deal Kit publish, a model). Drafts never train as gold without the owner's
  approval (``ml/c3/labels_from_corrections.py --approve``);
* the atom path, the way ``sow_fix_triage`` attaches its trace: the source line
  (file, where, text) -> the atom in the deal's latest envelope -> what the
  heads decided on it (``app.core.c3_sow_trace.trace_atoms`` plus whether the
  atom survived admission and which site it fed) -> the PM's verdict;
* ``skip`` when it cannot train: a routing probe, a disabled correction, no
  envelope, or no atom in the latest compile that matches it.

Read-only: it writes only under ``--out``. It never posts and never touches a DB.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import c3_sow_trace  # noqa: E402
from app.core.pm_feedback import HEAD_REGISTRY  # noqa: E402

#: Registry head (pm_feedback.HEAD_REGISTRY) -> (super head key, size). The
#: relation each one is stored under comes from the registry itself.
MAIN_HEADS: dict[str, tuple[str, str]] = {
    "type": ("col:label_type", "line"),
    "admission": ("col:admission", "line"),       # judgment alias -> col:admission
    "bom_owner": ("col:supplier", "line"),        # judgment alias -> col:supplier
    "site": ("jdg:site", "pair"),
    "site_role": ("jdg:site_role", "line"),
    "facility": ("jdg:facility", "line"),
    "conflict": ("jdg:conflict", "pair"),         # a "supports" verdict also trains rel:supports
}
#: Main heads with no PM correction route yet: nothing to queue.
NO_ROUTE = {
    "read:address_level": "no HEAD_REGISTRY head stores it; it is labeled only on the atom card",
}
RELATION_HEAD = {HEAD_REGISTRY[h].relation: h for h in MAIN_HEADS}

#: Authors that are not a person: their corrections are drafts.
_DRAFT = re.compile(r"claude|agent|assistant|\(bot\)|\bbot\b|model|judge|teacher|deal-?kit|publish|rule:|ollama",
                    re.I)
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_PROBE = re.compile(r"routing probe|synthetic, not a real atom", re.I)
#: decide() text v2 appends "[section: …] [intro: …]" after the atom's own words.
_DECIDE_TAIL = re.compile(r"\s\[(?:section|intro|doc|sheet|heading|row|table|page|ctx)\b[^\]]*\].*$", re.S)
_ARROW = re.compile(r":\s*(.*?)\s*(?:→|->)\s*")


def by_person(created_by: Any) -> bool:
    """A correction a person made (a PM through the UI, a labeler by email or
    user id). Agents, Deal Kit publishes and models write drafts."""
    v = str(created_by or "").strip()
    if not v or _DRAFT.search(v):
        return False
    return v.lower() == "pm" or "@" in v or bool(_UUID.match(v))


def was_of(instruction: str) -> str:
    """What the head said before, from "PM Keep / drop: deal_metadata → drop"."""
    m = _ARROW.search(str(instruction or ""))
    return m.group(1).strip() if m else ""


def exemplar_text(ex: str) -> str:
    """The atom's own words: no ``[ctx]`` line, no decide() v2 tail."""
    s = str(ex or "").split("\n[ctx]", 1)[0]
    return _DECIDE_TAIL.sub("", s).strip()


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", str(s or "").lower())).strip()


def load_corrections(path: str) -> list[dict[str, Any]]:
    """corrections.jsonl, or a folder of ``<id>.json`` blobs as downloaded."""
    p = Path(path).expanduser()
    if p.is_dir():
        files = sorted(glob.glob(str(p / "**" / "*.json"), recursive=True))
        return [json.loads(Path(f).read_text()) for f in files]
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


class Envelope:
    """One deal's latest envelope, indexed for finding a correction's atom."""

    def __init__(self, env: dict[str, Any]):
        self.env = env
        self.atoms = {str(a.get("id")): a for a in env.get("atoms") or [] if isinstance(a, dict)}
        self.suppressed = {str(a.get("id")): a for a in env.get("suppressed") or [] if isinstance(a, dict)}
        self.by_label_key: dict[str, str] = {}
        self.by_text: dict[str, list[str]] = {}
        for aid, a in self.atoms.items():
            lk = (a.get("structured") or {}).get("label_key") or a.get("label_key")
            if lk:
                self.by_label_key[str(lk)] = aid
            self.by_text.setdefault(_norm(a.get("text")), []).append(aid)
        self.sup_text: dict[str, list[str]] = {}
        for sid, a in self.suppressed.items():
            if a.get("label_key"):
                self.by_label_key.setdefault(str(a["label_key"]), sid)
            self.sup_text.setdefault(_norm(a.get("text")), []).append(sid)
        self.sites: dict[str, list[str]] = {}       # site name -> its source atoms
        self.site_of: dict[str, list[str]] = {}     # atom -> site names it fed
        for e in env.get("entities") or []:
            if not isinstance(e, dict) or e.get("entity_type") != "site":
                continue
            name = str(e.get("canonical_name") or "")
            src = [str(x) for x in e.get("source_atom_ids") or []]
            for n in {name, *[str(x) for x in e.get("aliases") or []]}:
                have = self.sites.setdefault(_norm(n), []) if _norm(n) else []
                have.extend(x for x in src if x not in have)
            for x in src:
                self.site_of.setdefault(x, []).append(name)

    @property
    def deal(self) -> str:
        name = str((self.env.get("crm") or {}).get("deal_name") or "")
        return name[:6] if name[:6].isdigit() else ""

    def find(self, atom_id: str = "", label_key: str = "", text: str = "") -> tuple[str | None, str]:
        """(atom id, how): by id, by the labeling page's key, by exact text,
        by the site name it fed. Re-parses rehash atom ids, so text is the
        fallback; an ambiguous text is not guessed."""
        aid = re.sub(r"^(site|atom):", "", str(atom_id or ""))
        if aid in self.atoms:
            return aid, "atom_id"
        if aid in self.suppressed:
            return aid, "suppressed"
        if label_key and label_key in self.by_label_key:
            hit = self.by_label_key[label_key]
            return hit, ("label_key" if hit in self.atoms else "suppressed")
        n = _norm(text)
        if not n:
            return None, "unresolved"
        hits = self.by_text.get(n, [])
        if len(hits) == 1:
            return hits[0], "text"
        if len(hits) > 1:
            return None, "ambiguous"
        sites = self.sites.get(n, [])
        if sites:
            return sites[0], "site_name"
        sup = self.sup_text.get(n, [])
        if len(sup) == 1:
            return sup[0], "suppressed"
        return None, "unresolved"

    def path(self, aid: str) -> dict[str, Any]:
        """Source line -> atom -> what the heads decided on it."""
        if aid in self.suppressed and aid not in self.atoms:
            s = self.suppressed[aid]
            return {"text": str(s.get("text") or "")[:600], "file": s.get("filename") or "",
                    "where": c3_sow_trace._where(s), "type": s.get("atom_type"),
                    "admitted": False, "dropped_by": {k: s.get(k) for k in ("stage", "reason", "kind", "emitted_by")
                                                      if s.get(k)}}
        row = dict(c3_sow_trace.trace_atoms(self.env, [aid]).get(aid) or {"missing": True})
        row["admitted"] = aid in self.atoms
        if self.site_of.get(aid):
            row["sites"] = self.site_of[aid]
        return row


def head_said(key: str, paths: list[dict[str, Any]]) -> Any:
    """What the heads (or the rule standing in for them) answered on this atom."""
    p = paths[0] if paths else {}
    if key == "col:label_type":
        return p.get("type")
    if key == "col:admission":
        return "keep" if p.get("admitted") else "drop"
    if key == "col:supplier":
        return (p.get("supplier") or [None])[0]
    if key in ("jdg:site_role", "jdg:facility"):
        return p.get("sites") or None
    if key == "jdg:site":
        a, b = ((paths + [{}, {}])[:2])
        sa, sb = set(a.get("sites") or []), set(b.get("sites") or [])
        if not sa or not sb:
            return None
        return "same_site" if sa & sb else "distinct_site"
    return None


def queue_rows(corr: dict[str, Any], envs: dict[str, Envelope]) -> list[dict[str, Any]]:
    head = RELATION_HEAD.get(str(corr.get("relation") or ""))
    if head is None:
        return []
    key, size = MAIN_HEADS[head]
    rel = corr.get("relations") or {}
    deals = [d for d in (rel.get("pm_seen_deals") or []) if d] or ([corr["scope_key"]] if corr.get("scope_key") else [])
    author = str(corr.get("created_by") or "")
    person = by_person(author)
    base = {
        "correction_id": corr.get("id"), "head": head, "key": key, "size": size,
        "relation": corr.get("relation"), "verdict": str(corr.get("verdict") or ""),
        "was": was_of(corr.get("instruction") or ""), "instruction": corr.get("instruction") or "",
        "scope": corr.get("scope") or "", "author": author, "by_person": person,
        "status": "person" if person else "draft",
        "at": float(corr.get("updated_at") or corr.get("created_at") or 0),
        "compile_id": rel.get("compile_id") or "", "label_key": rel.get("label_key") or "",
    }
    if head == "conflict" and base["verdict"] == "supports":
        base["also"] = ["rel:supports"]
    out = []
    exemplars = [e for e in corr.get("exemplars") or [] if str(e or "").strip()]
    for n, ex in enumerate(exemplars):
        text = exemplar_text(ex)
        row = dict(base, id=f"{corr.get('id')}#{n}", exemplar=text, deal_uuid=deals[0] if deals else "",
                   deal="", atoms=[], resolved=None, path=[], said=None, skip=None)
        if corr.get("status", "active") != "active":
            row["skip"] = "disabled"
        elif _PROBE.search(ex) or str(rel.get("atom_id") or "") == "routing_probe":
            row["skip"] = "probe"
        out.append(row)
        if row["skip"]:
            continue
        env = next((envs[d] for d in deals if d in envs), None)
        if env is None:
            row["skip"] = "no_envelope"
            continue
        row["deal_uuid"] = next(d for d in deals if d in envs)
        row["deal"] = env.deal
        if size == "pair":
            sides = [s.strip() for s in text.split("||")]
            if len(sides) != 2:
                row["skip"] = "not_a_pair"
                continue
            found = [env.find(text=s) for s in sides]
            row["resolved"] = "+".join(h for _, h in found)
            if any(a is None for a, _ in found) or found[0][0] == found[1][0]:
                row["skip"] = "unresolved"
                continue
            row["atoms"] = [a for a, _ in found]
            row["sides"] = sides
        else:
            # Only the first exemplar is the atom the correction names; a merged
            # correction's other exemplars are other atoms, found by text.
            aid, how = env.find(atom_id=(rel.get("atom_id") or corr.get("complaint_id") or "") if n == 0 else "",
                                label_key=row["label_key"] if n == 0 else "", text=text)
            row["resolved"] = how
            if aid is None:
                row["skip"] = "unresolved"
                continue
            row["atoms"] = [aid]
        row["path"] = [env.path(a) for a in row["atoms"]]
        row["said"] = head_said(key, row["path"])
    return out


def build(corrections: list[dict[str, Any]], envs: dict[str, Envelope]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    other = Counter()
    for c in corrections:
        got = queue_rows(c, envs)
        if not got:
            other[str(c.get("relation"))] += 1
        rows += got
    summary = {
        "corrections": len(corrections),
        "main_head_rows": len(rows),
        "trainable": sum(1 for r in rows if not r["skip"]),
        "by_head": {k: dict(Counter(("skip:" + r["skip"]) if r["skip"] else r["status"] for r in rows if r["key"] == k))
                    for k in sorted({r["key"] for r in rows})},
        "resolved_by": dict(Counter(r["resolved"] for r in rows if r["resolved"])),
        "not_main_heads": dict(other.most_common()),
        "no_route": NO_ROUTE,
        "deals": len({r["deal_uuid"] for r in rows if not r["skip"]}),
    }
    return rows, summary


def load_envelopes(folder: str, wanted: set[str]) -> dict[str, Envelope]:
    out = {}
    for d in sorted(wanted):
        p = Path(folder).expanduser() / f"{d}.json"
        if p.exists():
            out[d] = Envelope(json.loads(p.read_text()))
    return out


def wanted_deals(corrections: list[dict[str, Any]]) -> set[str]:
    out = set()
    for c in corrections:
        if str(c.get("relation")) not in RELATION_HEAD:
            continue
        rel = c.get("relations") or {}
        out.update(d for d in (rel.get("pm_seen_deals") or []) if d)
        if c.get("scope_key"):
            out.add(str(c["scope_key"]))
    return out


def report_md(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [f"# Main heads corrections queue ({summary['at']})", "",
             f"{summary['corrections']} corrections exported; {summary['main_head_rows']} main-head rows, "
             f"{summary['trainable']} trainable, across {summary['deals']} deals.", "",
             "| head | person | draft | skipped |", "|---|---|---|---|"]
    for k, c in summary["by_head"].items():
        skipped = ", ".join(f"{s[5:]} {n}" for s, n in sorted(c.items()) if s.startswith("skip:")) or "0"
        lines.append(f"| {k} | {c.get('person', 0)} | {c.get('draft', 0)} | {skipped} |")
    for k, why in summary["no_route"].items():
        lines.append(f"| {k} | - | - | {why} |")
    disagree = [r for r in rows if not r["skip"] and r["said"] not in (None, r["verdict"])][:20]
    if disagree:
        lines += ["", "## Where the head disagreed with the PM (first 20)", ""]
        for r in disagree:
            p = (r["path"] or [{}])[0]
            lines.append(f"- {r['deal'] or r['deal_uuid'][:8]} `{r['key']}` head said **{r['said']}**, "
                         f"{r['status']} says **{r['verdict']}**: {str(p.get('text') or r['exemplar'])[:120]!r} "
                         f"({p.get('file') or '?'} {p.get('where') or ''})")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corrections", required=True, help="corrections.jsonl or a folder of <id>.json blobs")
    ap.add_argument("--envelopes", required=True, help="folder of <deal uuid>.json latest envelopes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    corrections = load_corrections(a.corrections)
    if not corrections:
        print(f"FATAL main_heads_queue: no corrections in {a.corrections}", file=sys.stderr)
        return 3
    want = wanted_deals(corrections)
    envs = load_envelopes(a.envelopes, want)
    if want and not envs:
        print(f"FATAL main_heads_queue: none of {len(want)} deals has an envelope in {a.envelopes}", file=sys.stderr)
        return 3
    rows, summary = build(corrections, envs)
    summary["at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    summary["envelopes"] = f"{len(envs)}/{len(want)}"
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "queue.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "queue.md").write_text(report_md(summary, rows))
    print(f"queue: {summary['main_head_rows']} rows, {summary['trainable']} trainable, envelopes {summary['envelopes']} -> {out}")
    if not summary["trainable"]:
        print("FATAL main_heads_queue: no trainable main-head correction", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
