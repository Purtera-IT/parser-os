"""The description gate: a text change is a breaking change until the start weights say it is not.

The described heads score a line against the encoded text of each answer, so
rewriting an answer moves its prototype before any training step. Run E's
rewrite (b395fbb) took the live weights from 89 to 37 sites on 010358 with no
step taken; training brought it back only to 44 (pilot A root cause, M2). This
gate reads the START checkpoint twice in one process, the same weights and the
same lines, once under the texts it was trained on (``--old-texts``, e.g.
ml/c3/runE/init_texts.json, or ``--old-ref <commit>``) and once under this
code's texts, and fails when the text change alone costs any gated head more
than ``--max-drop`` (2 points):

* every head with at least ``--min-n`` labeled rows over the ``--deal`` files
  (exact match, the gate's own per-head score; ``--drafts-gold`` reads
  assistant drafts as gold, in memory, for holdout B);
* the share of ``--site-rows`` lines read as physical_site on each
  ``--site-dump`` (unlabeled dumps: 010358, 010447), CODE=REGEX pairs.

Under the old texts the lattice's space seeds are the old questions only
(a head added since INIT did not exist when INIT read); the new heads
themselves are not gated (INIT never learned them).

    python -m ml.c3.desc_gate --checkpoint $INIT --old-texts ml/c3/runE/init_texts.json \\
        --deal holdout/010038.json ... --site-dump 010358=dump_010358.json \\
        --site-rows '010358=^site_id: .*\\b7\\d{4}\\b' --json out.json

Exit 1 on a failure. Prints counts and keys only, never deal text.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
SITE = "physical_site"
TYPE_KEY = "col:label_type"

_TEXTS_AT = """
import json, sys
sys.path.insert(0, sys.argv[1])
from ml.c3.schema import load_schema
print(json.dumps([[o.key, o.description, [[a.value, a.description] for a in o.answers]]
                  for o in load_schema().opportunities], ensure_ascii=False))
"""


# ---------------------------------------------------------------- texts
def texts_at(ref: str, repo: str | Path | None = None) -> list:
    """[[key, question, [[value, answer text], ...]], ...] of the schema at commit ``ref``."""
    repo = Path(repo) if repo else REPO
    paths = ["ml/__init__.py", "ml/c3", "app/core/label_heads.json", "app/core/atom_types.json",
             "app/core/judgment_reasons.json"]
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "ref.tar"
        with open(archive, "wb") as fh:
            subprocess.run(["git", "-C", str(repo), "archive", ref, *paths], stdout=fh, check=True)
        root = Path(tmp) / "tree"
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        out = subprocess.run([sys.executable, "-I", "-c", _TEXTS_AT, str(root)], cwd=tmp,
                             capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def read_old_texts(old_texts: str = "", old_ref: str = "") -> list:
    if old_texts:
        return json.loads(Path(old_texts).read_text(encoding="utf-8"))["texts"]
    if old_ref:
        return texts_at(old_ref)
    raise SystemExit("--old-texts or --old-ref")


def schema_texts(schema) -> list:
    return [[o.key, o.description, [[a.value, a.description] for a in o.answers]] for o in schema.opportunities]


def changed_keys(schema, old: list) -> list[str]:
    """Keys INIT has whose question or answer texts differ here."""
    now = {k: (q, [list(a) for a in ans]) for k, q, ans in schema_texts(schema)}
    out = []
    for k, q, ans in old:
        if k in now and (now[k][0] != q or now[k][1] != [list(a) for a in ans]):
            out.append(k)
    return out


@torch.no_grad()
def old_description(model, schema, old: list) -> tuple[dict[str, Any], set[str]]:
    """``model.describe()`` with INIT's texts for INIT's keys (same answer values,
    same order); keys INIT lacks keep this code's text. Returns (desc, INIT keys)."""
    new = model.describe()
    by_key = {k: (q, ans) for k, q, ans in old}
    texts, slots = [], []
    for o in schema.opportunities:
        got = by_key.get(o.key)
        if got is None or [v for v, _ in got[1]] != [a.value for a in o.answers]:
            continue
        slots.append(o.key)
        texts.append(got[0])
        texts.extend(t for _, t in got[1])
    model.text.eval()
    emb = model.text(texts).float() if texts else None
    out, i = dict(new), 0
    for o in schema.opportunities:
        if o.key in slots:
            n = len(o.answers)
            q0, a0 = new[o.key]
            out[o.key] = (emb[i].to(q0.device, q0.dtype), emb[i + 1:i + 1 + n].to(a0.device, a0.dtype))
            i += 1 + n
    return out, set(slots)


@contextlib.contextmanager
def reading(model, desc: dict[str, Any], seed_keys: set[str] | None = None):
    """Read with ``desc``; with ``seed_keys``, the lattice's space and company
    seeds average only those keys' questions (the schema INIT read under)."""
    from . import model as M  # noqa: PLC0415

    saved_describe = model.__dict__.get("describe")
    saved_sd, saved_cd = M.space_descriptions, M.company_description
    model.describe = lambda: desc
    if seed_keys is not None:
        M.space_descriptions = lambda schema, d: saved_sd(schema, {k: v for k, v in d.items() if k in seed_keys})
        M.company_description = lambda schema, d: saved_cd(schema, {k: v for k, v in d.items() if k in seed_keys})
    try:
        yield
    finally:
        if saved_describe is None:
            del model.describe
        else:
            model.describe = saved_describe
        M.space_descriptions, M.company_description = saved_sd, saved_cd


# ---------------------------------------------------------------- reads
def score_parts(model, parts, device: str = "cpu") -> dict[str, dict[str, float]]:
    """Per-head exact over the parts' labeled rows, pooled by count."""
    from .loop import _score_oneshot  # noqa: PLC0415

    pooled: dict[str, list[float]] = {}
    for b in parts:
        for key, row in _score_oneshot(model, b, device).items():
            hit_n = pooled.setdefault(key, [0.0, 0.0])
            hit_n[0] += row["exact"] * row["n"]
            hit_n[1] += row["n"]
    return {k: {"exact": h / n, "n": n} for k, (h, n) in pooled.items() if n}


@torch.no_grad()
def site_share(model, parts, pattern: str) -> tuple[int, int]:
    """(lines read as physical_site, lines matching ``pattern``), first reading of each key."""
    rx = re.compile(pattern, re.I)
    model.eval()
    lt = model.schema.by_key()[TYPE_KEY]
    site = [a.value for a in lt.answers].index(SITE)
    seen: dict[str, bool] = {}
    for b in parts:
        out = model(b.oneshot(), company=b.company or None)
        pred = out.logits[TYPE_KEY].argmax(-1)
        for i, (k, t) in enumerate(zip(b.keys, b.texts)):
            if k not in seen and rx.search(str(t)):
                seen[k] = int(pred[i]) == site
    return sum(seen.values()), len(seen)


def failures(report: dict, max_drop: float, min_n: int) -> list[str]:
    bad = []
    for key, row in sorted(report.get("heads", {}).items()):
        if row["n"] >= min_n and row["new"] < row["old"] - max_drop - 1e-9:
            bad.append(f"{key}: {row['old']:.3f} -> {row['new']:.3f} (n={int(row['n'])})")
    for name, row in sorted(report.get("sites", {}).items()):
        if row["rows"] and (row["old"] - row["new"]) / row["rows"] > max_drop + 1e-9:
            bad.append(f"sites {name}: {row['old']} -> {row['new']} of {row['rows']}")
    return bad


def compare(model, schema, old: list, labeled: list, dumps: dict[str, tuple[list, str]],
            device: str = "cpu", old_model=None, new_seeds: str = "all") -> dict:
    """``old_model`` (default ``model``) is read under the old texts: a refit start
    (head_fit on the new texts) is compared with the checkpoint it was refit from."""
    old_model = old_model if old_model is not None else model
    new_desc = model.describe()
    old_desc, init_keys = old_description(old_model, schema, old)
    report: dict[str, Any] = {"changed": changed_keys(schema, old), "heads": {}, "sites": {}}
    with reading(old_model, old_desc, init_keys):
        old_scores = score_parts(old_model, labeled, device) if labeled else {}
        old_sites = {n: site_share(old_model, p, rx) for n, (p, rx) in dumps.items()}
    # new_seeds="init": the new texts with the lattice seeds of INIT's heads only, which
    # separates a rewrite of a head's own text from the new heads' questions moving a
    # space's seed (a diagnostic; the gate itself reads "all", what the run reads).
    with reading(model, new_desc, init_keys if new_seeds == "init" else None):
        new_scores = score_parts(model, labeled, device) if labeled else {}
        new_sites = {n: site_share(model, p, rx) for n, (p, rx) in dumps.items()}
    for key in sorted(set(old_scores) & set(new_scores)):
        if key not in init_keys:
            continue
        report["heads"][key] = {"old": old_scores[key]["exact"], "new": new_scores[key]["exact"],
                                "n": old_scores[key]["n"]}
    for n in dumps:
        report["sites"][n] = {"old": old_sites[n][0], "new": new_sites[n][0], "rows": old_sites[n][1]}
    return report


# ---------------------------------------------------------------- load
def _text_cache(model, path: str) -> None:
    """Line vectors of the frozen encoder by sha1 of the text (the root-cause probe's cache);
    ``model.save_text_cache()`` writes the new ones back."""
    disk = torch.load(path) if Path(path).exists() else {}
    start = len(disk)

    def save():
        if len(disk) > start:
            tmp = f"{path}.tmp"
            torch.save(disk, tmp)
            Path(tmp).replace(path)
    model.save_text_cache = save
    enc = model.text
    orig = enc.forward

    def key(t):
        return hashlib.sha1(t.encode()).hexdigest()

    def forward(texts):
        texts = [str(t) for t in texts]
        miss = sorted({t for t in texts if key(t) not in disk})
        for a in range(0, len(miss), 64):
            part = miss[a:a + 64]
            for t, v in zip(part, orig(part).float()):
                disk[key(t)] = v.detach().clone()
        return torch.stack([disk[key(t)] for t in texts]).float()

    enc.forward = forward
    if hasattr(enc, "fixed"):
        enc.fixed = forward


def load_start(checkpoint: str, encoder: str, init_shape: str, device: str = "cpu", text_cache: str = ""):
    """INIT under this code: text changes allowed, the heads added since left at their init."""
    from .gates import load_checkpoint, schema_digests_at  # noqa: PLC0415
    from .loop import SMALL, live_student, without_heads  # noqa: PLC0415
    from .model import C3Model  # noqa: PLC0415
    from .new_heads import derive  # noqa: PLC0415
    from .readiness import digest  # noqa: PLC0415
    from .schema import load_schema, shape_digest  # noqa: PLC0415

    schema = load_schema()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    tiny = "text.bag.weight" in payload["state"]
    del payload
    shape = json.loads(Path(init_shape).read_text(encoding="utf-8"))["shape"]
    added = ",".join(derive(schema, [(k, list(v)) for k, v in shape]))
    full, shp = schema_digests_at(init_shape)
    model = C3Model(schema, SMALL) if tiny else live_student(schema, encoder)
    load_checkpoint(checkpoint, model, schema_sha256=digest([asdict(o) for o in schema.opportunities]),
                    allow_text_change=True, shape_sha256=shape_digest(without_heads(schema, added)),
                    known_shapes={full: shp})
    if device == "cpu" and not tiny:
        body = model.text._modules.pop("model", None)
        model.float()
        if body is not None:
            model.text._modules["model"] = body
    model.to(device).eval()
    if text_cache:
        _text_cache(model, text_cache)
    return model, schema


def _parts(path: str, schema, drafts_gold: bool, max_atoms: int, lines_only: bool):
    from .data import DealExample  # noqa: PLC0415
    from .loop import load_deal, training_batches  # noqa: PLC0415
    from .predict import deal_parts  # noqa: PLC0415

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if lines_only or not isinstance(payload.get("labels"), list):
        for a in payload["atoms"]:
            a.setdefault("key", a.get("label_key") or a.get("atom_id"))
        parts, _ = deal_parts(DealExample.from_dict(payload), schema, max_atoms, lines_only=True)
        return parts
    if drafts_gold:
        from .data import _is_a_person  # noqa: PLC0415
        for part in ("labels", "judgments", "deal_answers", "links", "missed_history"):
            for row in payload.get(part) or []:
                if isinstance(row, dict) and not _is_a_person(row.get("labeler")):
                    row["labeler"] = "gate-draft-gold"
        with tempfile.NamedTemporaryFile("w", suffix=".json", dir=Path(path).parent, delete=True) as fh:
            json.dump(payload, fh)
            fh.flush()
            deal = load_deal(fh.name)
    else:
        deal = load_deal(path)
    parts, _ = training_batches(deal, schema, max_atoms=max_atoms)
    return parts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="the start checkpoint (INIT)")
    ap.add_argument("--old-checkpoint", default="",
                    help="read the old texts with this checkpoint instead (the one --checkpoint was refit from)")
    ap.add_argument("--old-texts", default=str(REPO / "ml/c3/runE/init_texts.json"))
    ap.add_argument("--old-ref", default="", help="instead of --old-texts: the commit INIT was trained on")
    ap.add_argument("--init-shape", default=str(REPO / "ml/c3/runE/init_shape.json"))
    ap.add_argument("--old-init-shape", default="",
                    help="shape for --old-checkpoint when it differs from INIT's (default: --init-shape)")
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    ap.add_argument("--deal", nargs="*", default=[], help="labeled deal files (training format)")
    ap.add_argument("--drafts-gold", nargs="*", default=[], help="labeled files whose drafts read as gold")
    ap.add_argument("--site-dump", action="append", default=[], help="CODE=path of an unlabeled dump")
    ap.add_argument("--site-rows", action="append", default=[], help="CODE=regex of its site lines")
    ap.add_argument("--max-atoms", type=int, default=8192)
    ap.add_argument("--max-drop", type=float, default=0.02)
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--text-cache", default="")
    ap.add_argument("--graph-set", default="",
                        help="graph flags over the checkpoint's (graph.parse_graph keys), e.g. lattice_doc=1: "
                             "read both texts with the run's graph")
    ap.add_argument("--new-seeds", choices=("all", "init"), default="all",
                    help="diagnostic: 'init' reads the new texts with INIT's heads' lattice seeds only")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    old = read_old_texts("" if args.old_ref else args.old_texts, args.old_ref)
    model, schema = load_start(args.checkpoint, args.encoder, args.init_shape, args.device, args.text_cache)
    if args.graph_set:
        from .graph import GRAPH_DEFAULTS, parse_graph  # noqa: PLC0415

        over = parse_graph(args.graph_set)
        model.enable_graph({**model.graph, **{k: v for k, v in over.items() if v != GRAPH_DEFAULTS[k]}})
        model.to(args.device).eval()
        print("graph " + " ".join(f"{k}={v}" for k, v in model.graph.items()), flush=True)
    labeled = []
    for p in args.deal:
        labeled += _parts(p, schema, False, args.max_atoms, False)
    for p in args.drafts_gold:
        labeled += _parts(p, schema, True, args.max_atoms, False)
    rows = dict(s.split("=", 1) for s in args.site_rows)
    dumps = {}
    for spec in args.site_dump:
        code, _, path = spec.partition("=")
        if code not in rows:
            raise SystemExit(f"--site-dump {code}: no --site-rows {code}=REGEX")
        dumps[code] = (_parts(path, schema, False, args.max_atoms, True), rows[code])
    old_model = None
    if args.old_checkpoint:
        old_model, _ = load_start(args.old_checkpoint, args.encoder, args.old_init_shape or args.init_shape, args.device, "")
        if args.text_cache:
            # The same frozen encoder weights: share the cached line vectors.
            old_model.text.forward = model.text.forward
            if hasattr(old_model.text, "fixed"):
                old_model.text.fixed = model.text.forward
        old_model.enable_graph(model.graph)
        old_model.to(args.device).eval()
    report = compare(model, schema, old, labeled, dumps, args.device, old_model, args.new_seeds)
    if hasattr(model, "save_text_cache"):
        model.save_text_cache()
    bad = failures(report, args.max_drop, args.min_n)
    report["failed"] = bad
    print(f"description gate: {len(report['changed'])} heads' texts differ from INIT's; "
          f"{len(labeled)} labeled parts, {len(dumps)} site dumps")
    for key, row in sorted(report["heads"].items(), key=lambda kv: kv[1]["new"] - kv[1]["old"]):
        if row["n"] >= args.min_n and abs(row["new"] - row["old"]) > 1e-9:
            print(f"  {key:34} old {row['old']:.3f} new {row['new']:.3f} "
                  f"({row['new'] - row['old']:+.3f}) n={int(row['n'])}")
    for name, row in sorted(report["sites"].items()):
        print(f"  sites {name}: old {row['old']} new {row['new']} of {row['rows']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    if bad:
        print("DESCRIPTION GATE FAILED (the text change alone): " + "; ".join(bad))
        return 1
    print("description gate: pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
