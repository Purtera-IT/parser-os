"""The heads a checkpoint does not have, derived from the schema diff (runE.sh NEW_HEADS).

A run that starts from an older checkpoint (``loop.py --init-from
--allow-text-change --allow-new-heads K,...``) may only *add* heads: the
checkpoint's opportunity keys, their order and every answer-value list must
be what this schema has once the added heads are left out
(``schema.shape_digest(loop.without_heads(schema, added))``). Listing the
added heads by hand drifted (read:feeds_decision was missing, and the INIT load
failed with "differs in more than text"), so the launch derives them:

* the old shape is the schema at ``--ref`` (the commit INIT was trained on,
  runE.sh INIT_SCHEMA_REF; its own registries and schema code, as
  ``gates.schema_digests_at`` reads them), or a saved shape file
  (``--shape``, ``ml/c3/runE/init_shape.json`` for the live run C checkpoint:
  ``{"ref", "schema_sha256", "shape": [[key, [answer values]], ...]}``, keys and
  answer values only), for a pod copy of the repo with no git history;
* the added heads are this schema's keys the old shape lacks;
* anything else (a key removed, keys reordered, an answer list changed) is
  not a head addition: the tool fails and names each difference;
* with ``--checkpoint`` the old shape's digest must be the one the checkpoint
  stored (``shape_sha256``), or its schema digest the ref's (a checkpoint saved
  before shape digests): the ref really is the checkpoint's schema.

    python -m ml.c3.new_heads --ref 0b6efce --checkpoint $INIT     # prints k1,k2,...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_SHAPE_AT = """
import json, sys
sys.path.insert(0, sys.argv[1])
from ml.c3.schema import load_schema
print(json.dumps([(o.key, [a.value for a in o.answers]) for o in load_schema().opportunities]))
"""


def _digest(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def shape_at(ref: str, repo: str | Path | None = None) -> list[tuple[str, list[str]]]:
    """``schema.shape_of`` of the schema at commit ``ref`` (its own code and registries)."""
    import subprocess  # noqa: PLC0415
    import tarfile  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

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
        out = subprocess.run([sys.executable, "-I", "-c", _SHAPE_AT, str(root)], cwd=tmp,
                             capture_output=True, text=True, check=True).stdout
    return [(k, list(v)) for k, v in json.loads(out)]


def diff(new: list[tuple[str, list[str]]], old: list[tuple[str, list[str]]]) -> tuple[list[str], list[str]]:
    """(keys added, differences that are not a head addition)."""
    old_keys = [k for k, _ in old]
    have = set(old_keys)
    added = [k for k, _ in new if k not in have]
    kept = [(k, list(v)) for k, v in new if k in have]
    problems = []
    new_keys = {k for k, _ in new}
    problems += [f"{k}: removed (INIT has it, this schema does not)" for k in old_keys if k not in new_keys]
    old_by = {k: list(v) for k, v in old}
    for k, v in kept:
        if v != old_by[k]:
            gone = [a for a in old_by[k] if a not in v]
            extra = [a for a in v if a not in old_by[k]]
            what = []
            if extra:
                what.append("answers added " + ", ".join(extra))
            if gone:
                what.append("answers removed " + ", ".join(gone))
            if not what:
                what.append("answers reordered")
            problems.append(f"{k}: {'; '.join(what)}")
    order = [k for k, _ in kept]
    if not problems and order != [k for k in old_keys if k in new_keys]:
        moved = next(k for k, o in zip(order, [k for k in old_keys if k in new_keys]) if k != o)
        problems.append(f"keys reordered (first at {moved})")
    return added, problems


def derive(schema, old: list[tuple[str, list[str]]]) -> list[str]:
    """The heads ``schema`` adds to ``old``; SystemExit naming every other difference."""
    from .loop import without_heads  # noqa: PLC0415
    from .schema import shape_digest, shape_of  # noqa: PLC0415

    added, problems = diff(shape_of(schema), old)
    if problems:
        raise SystemExit("the schema differs from INIT's in more than added heads:\n  "
                         + "\n  ".join(problems))
    if shape_digest(without_heads(schema, ",".join(added))) != _digest([list(x) for x in old]):
        raise SystemExit("the schema without the added heads is not INIT's shape")
    return added


def checkpoint_matches(path: str, old: list[tuple[str, list[str]]], ref: str = "",
                       schema_sha256: str = "") -> str:
    """'' when the checkpoint was saved under ``old`` (its shape digest, or the
    ref's schema digest: ``schema_sha256``, else read at ``ref``), else why not."""
    import torch  # noqa: PLC0415

    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    stored = payload.get("shape_sha256")
    want = _digest([list(x) for x in old])
    if stored:
        return "" if stored == want else f"{path} was saved under shape {stored[:12]}, not {want[:12]} (the ref's)"
    if not schema_sha256 and ref:
        from .gates import schema_digests_at  # noqa: PLC0415

        schema_sha256, _ = schema_digests_at(ref)
    if schema_sha256 and payload.get("schema_sha256") == schema_sha256:
        return ""
    return f"{path} carries no shape digest and its schema digest is not the ref's"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ref", default="", help="the commit INIT was trained on (INIT_SCHEMA_REF)")
    src.add_argument("--shape", default="", help="a saved shape file (--write-shape)")
    p.add_argument("--checkpoint", default="", help="INIT: its stored shape must be the ref's")
    p.add_argument("--write-shape", default="", help="with --ref: save the ref's shape file here")
    args = p.parse_args(argv)
    from .schema import load_schema  # noqa: PLC0415

    full = ""
    if args.ref:
        old = shape_at(args.ref)
        if args.write_shape:
            from .gates import schema_digests_at  # noqa: PLC0415

            full, _ = schema_digests_at(args.ref)
            Path(args.write_shape).write_text(json.dumps(
                {"ref": args.ref, "schema_sha256": full, "shape_sha256": _digest([list(x) for x in old]),
                 "shape": old}) + "\n", encoding="utf-8")
    else:
        blob = json.loads(Path(args.shape).read_text(encoding="utf-8"))
        old = [(k, list(v)) for k, v in blob["shape"]]
        full = str(blob.get("schema_sha256") or "")
    if args.checkpoint:
        why = checkpoint_matches(args.checkpoint, old, args.ref, full)
        if why:
            raise SystemExit(f"INIT is not the ref's schema: {why}")
    added = derive(load_schema(), old)
    print(",".join(added))
    print(f"new heads since {args.ref or args.shape}: {len(added)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
