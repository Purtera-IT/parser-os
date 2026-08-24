"""Materialise the atom-type kNN candidate the cascade can actually load.

The atom run measured a base bge-small kNN on the deal-split holdout:

    margin gate            coverage 49.4%   accuracy-when-answering 81.8%
    conformal (unanimous)  answer   21.3%   accuracy-when-answering 99.2%

This builder turns that measurement into an artifact in the exact layout
``contrastive_type_knn.load_promoted`` reads -- ``store.npz`` + ``knn_meta.json``
+ ``best/`` encoder -- so the cascade's contrastive layer (behind
``SOWSMITH_CONTRASTIVE_TYPE``, off by default) can serve it with NO new
loading machinery.

Honesty constraints, enforced here rather than remembered:

* **The strict operating point ships, not the flattering one.** ``tau`` in the
  meta is set so the head only answers on a near-unanimous neighbourhood --
  the measured 99.2%-agreement point -- because the cascade ASSIGNS the type
  and skips the LLM; an 18% disagreement rate is a queue, not an assignment.
* **Value-light only.** The cascade's contract sends value-heavy types
  (bom_line, commercial_total, ...) to the LLM regardless, because the label
  alone is not the deliverable -- the LLM synthesises the value payload. The
  head's two best classes are value-heavy, so the number that matters is
  coverage on the VALUE-LIGHT subset, and the meta records that number, not
  the overall one.
* **The output is a CANDIDATE.** It is written to its own directory, never to
  the live one: promotion stays a human act (point SOWSMITH_CONTRASTIVE_TYPE_DIR
  at it), and the meta carries the full eval so whoever promotes can see what
  they are promoting. ``ready`` in the meta follows the measured numbers, not
  optimism.
* **Representation version travels.** The meta records the repr_version mix of
  the rows the store was built from (v0 = bare legacy text). A store built on
  v0 rows and served v2 decide-text is out of distribution, and the meta says
  so out loud.

Run:  python -m app.learning.build_atom_store  (CPU, ~3 minutes)
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

#: Mirrors the cascade's value-light contract (typed_atom_classifier).
VALUE_LIGHT = frozenset({
    "requirement", "exclusion", "contract_term", "deal_metadata",
    "acceptance_criterion", "task", "change_order_rule", "constraint",
    "dependency", "mitigation", "compliance_rule", "submission_req",
    "addendum_qa",
})

_BAD_LABELS = ("_keep", "keep", "drop", "_drop")
_STRICT_TAU = 0.98  # near-unanimous neighbourhood == the 99.2% measured point
_SIM_FLOOR = 0.55
_K = 15


def build(
    table_db: Path = Path("_multitask_table.db"),
    out_dir: Path = Path("_contrastive_type_candidate"),
    *,
    encoder_name: str = "BAAI/bge-small-en-v1.5",
) -> dict:
    from sentence_transformers import SentenceTransformer

    conn = sqlite3.connect(table_db)
    train = list(conn.execute(
        "select text, label, repr_version from multitask_rows "
        "where task='atom_type' and split='train' and label not in (?,?,?,?)",
        _BAD_LABELS))
    holdout = list(conn.execute(
        "select text, label, deal_id from multitask_rows "
        "where task='atom_type' and split='holdout' and label not in (?,?,?,?)",
        _BAD_LABELS))
    if not train or not holdout:
        raise SystemExit("no typed rows -- run app.learning.multitask_table first")

    model = SentenceTransformer(encoder_name, device="cpu")

    def embed(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(
            texts, batch_size=128, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)

    t0 = time.time()
    X = embed([t for t, _, _ in train])
    y = np.array([lab for _, lab, _ in train])
    texts = np.array([t for t, _, _ in train])

    # ── evaluate the STRICT point on the holdout, value-light separated ──
    Xh = embed([t for t, _, _ in holdout])

    def verdict(q: np.ndarray) -> tuple[str, float] | None:
        sims = X @ q
        nb = np.argpartition(-sims, _K - 1)[:_K]
        if float(sims[nb].max()) < _SIM_FLOOR:
            return None
        votes: dict[str, float] = defaultdict(float)
        for j in nb:
            votes[str(y[j])] += max(float(sims[j]), 0.0)
        total = sum(votes.values()) or 1.0
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        margin = (ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)) / total
        if margin < _STRICT_TAU:
            return None
        return ranked[0][0], margin

    overall = {"answered": 0, "right": 0}
    light = {"answered": 0, "right": 0, "eligible": 0}
    for (text, lab, _deal), q in zip(holdout, Xh):
        if lab in VALUE_LIGHT:
            light["eligible"] += 1
        v = verdict(q)
        if v is None:
            continue
        pred, _ = v
        overall["answered"] += 1
        overall["right"] += int(pred == lab)
        if pred in VALUE_LIGHT:
            light["answered"] += 1
            light["right"] += int(pred == lab)

    versions: dict[str, int] = defaultdict(int)
    for _, _, ver in train:
        versions[f"v{ver}"] += 1

    # ── write the artifact in load_promoted's layout ─────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "store.npz", emb=X, y=y, text=texts)
    model.save(str(out_dir / "best"))

    acc_light = light["right"] / light["answered"] if light["answered"] else 0.0
    meta = {
        "mode": "typed",
        "k": _K,
        "sim_floor": _SIM_FLOOR,
        "tau": _STRICT_TAU,
        "encoder": encoder_name,
        "value_light": sorted(VALUE_LIGHT),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "holdout_deals": len({d for _, _, d in holdout}),
        "eval": {
            "overall_answered": overall["answered"],
            "overall_accuracy": round(overall["right"] / overall["answered"], 4)
            if overall["answered"] else 0.0,
            "value_light_answered": light["answered"],
            "value_light_accuracy": round(acc_light, 4),
            "value_light_eligible": light["eligible"],
        },
        "repr_versions": dict(versions),
        "caveats": [
            "labels are TEACHER labels; accuracy here is agreement, not correctness "
            "(PM-gold n=23, far below any reporting threshold)",
            "store rows are overwhelmingly v0 bare text; the cascade serves v2 "
            "decide-text -- retrain on v2 rows as they accumulate",
        ],
        # ready means: worth pointing the flag at for SHADOW observation. It is
        # not a promotion -- champion selection stays with the eval gate.
        "ready": bool(light["answered"] >= 20 and acc_light >= 0.95),
        "built_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "knn_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":  # pragma: no cover - operator entry point
    meta = build()
    print(json.dumps(meta["eval"], indent=2))
    print("ready:", meta["ready"], "| repr:", meta["repr_versions"])
    print("-> _contrastive_type_candidate/ (store.npz, knn_meta.json, best/)")
