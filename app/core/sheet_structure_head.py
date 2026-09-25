"""Structural sheet head (PUR-21 / PUR-51 / PUR-52).

The rule classifier in ``app.parsers.sheet_classifier`` recognises a handful of
sheet kinds from word lists. Everything it does not recognise used to fall
through to SCOPE. This head learns what a worksheet IS from hand labels, keyed
on STRUCTURE only:

* header tokens (hashed bag of words from the detected header row),
* per-column type profile (numeric / money / date / text / blank / id-like),
* row shape (row count, width, fill density, header position).

The sheet NAME is deliberately excluded: tab names are renamed, copied between
templates and forged far more often than a table's shape changes, and a head
keyed on names would learn deal-specific coincidences.

Contracts (same spirit as ``admission_head``):

* **Pure + portable.** Training is a small numpy softmax regression; a saved
  head is JSON (weights, bias, classes, threshold). No sklearn to serve.
* **Abstain by default.** No labels file / no saved head -> ``predict`` returns
  ``None``. Below the confidence threshold -> ``None``. The classifier turns
  ``None`` into the explicit UNCLASSIFIED state, never into SCOPE.
* **Synthetic only in tests.** Nothing here reads data on its own; callers
  supply rows and labels.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# PUR-51 label vocabulary. Closed set: a labels file carrying anything else is
# rejected at load time rather than silently training a new class.
LABEL_CLASSES: tuple[str, ...] = (
    "scope",
    "pricing",
    "bill_of_materials",
    "site_list",
    "schedule",
    "contact_list",
    "boilerplate",
    "junk",
)

# Label -> SheetRole value used by the parsers' router. Only kinds whose rows
# genuinely describe work are routed to SCOPE; pricing goes to COMMERCIAL;
# contact lists / boilerplate / junk are retained as markers (DROP), never mined.
# `schedule` is mined as scope rows (milestones/dates) — revisit once labels
# show whether schedule sheets deserve their own emitter.
LABEL_TO_ROLE: dict[str, str] = {
    "scope": "scope",
    "bill_of_materials": "scope",
    "site_list": "scope",
    "schedule": "scope",
    "pricing": "catalog",
    "contact_list": "reference",
    "boilerplate": "instructions",
    "junk": "reference",
}

DEFAULT_THRESHOLD = 0.6
HEAD_ENV = "SHEET_STRUCTURE_HEAD_PATH"
DEFAULT_HEAD_PATH = Path(__file__).parent / "data" / "sheet_structure_head.json"

_HASH_DIM = 256
_TYPE_KINDS = ("numeric", "money", "date", "text", "idlike", "blank")
_MONEY_RE = re.compile(r"^\s*[-(]?\s*[$€£]\s*[\d,]+(\.\d+)?\)?\s*$")
_NUM_RE = re.compile(r"^\s*[-+(]?[\d,]*\.?\d+\)?%?\s*$")
_DATE_RE = re.compile(
    r"^\s*(\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*$"
)
_IDLIKE_RE = re.compile(r"^[A-Za-z]{0,6}[-_]?\d{2,}[A-Za-z0-9\-_]*$")
_TOKEN_RE = re.compile(r"[a-z]+")


# ── structural profile ──────────────────────────────────────────────


def _s(c: Any) -> str:
    return "" if c is None else str(c).strip()


def cell_kind(c: Any) -> str:
    if c is None:
        return "blank"
    if isinstance(c, bool):
        return "text"
    if isinstance(c, (int, float)):
        return "numeric"
    if hasattr(c, "year") and hasattr(c, "month"):
        return "date"
    s = _s(c)
    if not s:
        return "blank"
    if _MONEY_RE.match(s):
        return "money"
    if _DATE_RE.match(s):
        return "date"
    if _NUM_RE.match(s):
        return "numeric"
    if _IDLIKE_RE.match(s):
        return "idlike"
    return "text"


def detect_header_row(rows: list[list[Any]]) -> int | None:
    """First row at least 60% as wide as the widest (min 2 cells), mostly text."""
    widths = [sum(1 for c in r if _s(c)) for r in rows[:50]]
    if not widths or max(widths) < 2:
        return None
    widest = max(widths)
    for i, w in enumerate(widths):
        if w >= max(2, int(widest * 0.6)):
            kinds = [cell_kind(c) for c in rows[i] if _s(c)]
            if kinds and sum(k == "text" for k in kinds) / len(kinds) >= 0.5:
                return i
    return None


@dataclass
class SheetProfile:
    """Structure of one sheet. Contains no cell values beyond header text."""

    headers: list[str]
    header_row: int | None
    row_count: int
    nonblank_rows: int
    max_width: int
    mean_width: float
    fill_density: float
    column_types: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "headers": self.headers,
            "header_row": self.header_row,
            "row_count": self.row_count,
            "nonblank_rows": self.nonblank_rows,
            "max_width": self.max_width,
            "mean_width": round(self.mean_width, 3),
            "fill_density": round(self.fill_density, 3),
            "column_types": self.column_types,
        }


def profile_sheet(rows: list[list[Any]], *, max_scan_rows: int = 2000) -> SheetProfile:
    rows = rows or []
    nonblank = [r for r in rows if any(_s(c) for c in r)]
    widths = [sum(1 for c in r if _s(c)) for r in nonblank]
    max_width = max(widths) if widths else 0
    hdr = detect_header_row(rows)
    headers = [_s(c) for c in rows[hdr]] if hdr is not None else []
    while headers and not headers[-1]:
        headers.pop()
    ncols = max(max_width, len(headers))
    body = rows[(hdr + 1) if hdr is not None else 0 :][:max_scan_rows]
    col_types: list[dict[str, float]] = []
    for ci in range(min(ncols, 40)):
        counts = {k: 0 for k in _TYPE_KINDS}
        for r in body:
            counts[cell_kind(r[ci] if ci < len(r) else None)] += 1
        n = max(1, len(body))
        col_types.append({k: round(v / n, 3) for k, v in counts.items()})
    span = max(1, max((len(r) for r in nonblank), default=1))
    filled = sum(widths)
    density = filled / (len(nonblank) * span) if nonblank else 0.0
    return SheetProfile(
        headers=headers,
        header_row=hdr,
        row_count=len(rows),
        nonblank_rows=len(nonblank),
        max_width=max_width,
        mean_width=(sum(widths) / len(widths)) if widths else 0.0,
        fill_density=density,
        column_types=col_types,
    )


def _hash_token(tok: str) -> int:
    return int.from_bytes(hashlib.sha1(tok.encode()).digest()[:4], "big") % _HASH_DIM


FEATURE_DIM = _HASH_DIM + 2 * len(_TYPE_KINDS) + 8


def featurize(profile: SheetProfile) -> np.ndarray:
    """Structure -> fixed vector. Sheet name is never an input."""
    v = np.zeros(FEATURE_DIM, dtype=np.float64)
    toks: list[str] = []
    for h in profile.headers:
        toks.extend(_TOKEN_RE.findall(h.lower()))
    for t in toks:
        v[_hash_token(t)] += 1.0
    if toks:
        norm = np.linalg.norm(v[:_HASH_DIM])
        if norm:
            v[:_HASH_DIM] /= norm
    o = _HASH_DIM
    cols = profile.column_types or []
    if cols:
        for k_i, k in enumerate(_TYPE_KINDS):
            vals = [c.get(k, 0.0) for c in cols]
            v[o + k_i] = float(np.mean(vals))  # mean fraction of this kind
            v[o + len(_TYPE_KINDS) + k_i] = sum(1 for x in vals if x >= 0.5) / len(cols)  # share of columns dominated
    o += 2 * len(_TYPE_KINDS)
    v[o] = math.log1p(profile.nonblank_rows) / 10.0
    v[o + 1] = math.log1p(profile.max_width) / 4.0
    v[o + 2] = profile.mean_width / max(1.0, profile.max_width)
    v[o + 3] = profile.fill_density
    v[o + 4] = 1.0 if profile.header_row is not None else 0.0
    v[o + 5] = min(1.0, (profile.header_row or 0) / 20.0)
    v[o + 6] = 1.0 if profile.max_width <= 1 else 0.0
    v[o + 7] = 1.0  # bias-like constant
    return v


# ── head ────────────────────────────────────────────────────────────


@dataclass
class SheetPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]

    @property
    def role(self) -> str:
        return LABEL_TO_ROLE[self.label]


@dataclass
class SheetStructureHead:
    classes: list[str]
    weights: np.ndarray  # (C, D)
    bias: np.ndarray  # (C,)
    threshold: float = DEFAULT_THRESHOLD
    n_train: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def proba(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.weights.T + self.bias
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def predict_rows(self, rows: list[list[Any]]) -> SheetPrediction | None:
        return self.predict_profile(profile_sheet(rows))

    def predict_profile(self, profile: SheetProfile) -> SheetPrediction | None:
        """``None`` = abstain (below threshold)."""
        p = self.proba(featurize(profile)[None, :])[0]
        i = int(np.argmax(p))
        probs = {c: round(float(p[j]), 4) for j, c in enumerate(self.classes)}
        if float(p[i]) < self.threshold:
            return None
        return SheetPrediction(self.classes[i], float(p[i]), probs)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "sheet_structure_head",
            "version": 1,
            "feature_dim": FEATURE_DIM,
            "classes": self.classes,
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "threshold": self.threshold,
            "n_train": self.n_train,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "SheetStructureHead":
        if d.get("kind") != "sheet_structure_head" or int(d.get("feature_dim", -1)) != FEATURE_DIM:
            raise ValueError("incompatible sheet_structure_head")
        classes = list(d["classes"])
        bad = [c for c in classes if c not in LABEL_CLASSES]
        if bad:
            raise ValueError(f"unknown classes in head: {bad}")
        return cls(
            classes=classes,
            weights=np.asarray(d["weights"], dtype=np.float64),
            bias=np.asarray(d["bias"], dtype=np.float64),
            threshold=float(d.get("threshold", DEFAULT_THRESHOLD)),
            n_train=int(d.get("n_train", 0)),
            meta=dict(d.get("meta") or {}),
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json()))


def fit_sheet_head(
    X: np.ndarray,
    y: list[str],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    l2: float = 1e-3,
    epochs: int = 400,
    lr: float = 0.5,
    seed: int = 0,
) -> SheetStructureHead:
    """Softmax regression by full-batch gradient descent (deterministic)."""
    bad = sorted({lab for lab in y if lab not in LABEL_CLASSES})
    if bad:
        raise ValueError(f"labels outside class set: {bad}")
    classes = [c for c in LABEL_CLASSES if c in set(y)]
    if len(classes) < 2:
        raise ValueError("need at least two distinct labels to train")
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(y), len(classes)))
    for r, lab in enumerate(y):
        Y[r, idx[lab]] = 1.0
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, (len(classes), X.shape[1]))
    b = np.zeros(len(classes))
    n = max(1, len(y))
    for _ in range(epochs):
        z = X @ W.T + b
        z -= z.max(axis=1, keepdims=True)
        P = np.exp(z)
        P /= P.sum(axis=1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (G.T @ X + l2 * W)
        b -= lr * G.sum(axis=0)
    return SheetStructureHead(classes, W, b, threshold=threshold, n_train=len(y))


# ── labels file ─────────────────────────────────────────────────────


def read_labels(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read the labelling JSONL written by ``scripts/sheet_fallthrough_labels.py``.

    Rows with an empty ``label`` are skipped (not yet labelled). A label outside
    :data:`LABEL_CLASSES` raises — a typo must not become a class.
    """
    out: list[dict[str, Any]] = []
    p = Path(path)
    if p.suffix.lower() == ".csv":
        import csv

        with p.open(newline="") as fh:
            recs = []
            for r in csv.DictReader(fh):
                r = dict(r)
                for k in ("headers", "column_types"):
                    if isinstance(r.get(k), str) and r[k].startswith(("[", "{")):
                        r[k] = json.loads(r[k])
                recs.append(r)
    else:
        recs = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    for r in recs:
        lab = str(r.get("label") or "").strip()
        if not lab:
            continue
        if lab not in LABEL_CLASSES:
            raise ValueError(f"label {lab!r} for {r.get('sheet_id')} not in {LABEL_CLASSES}")
        r["label"] = lab
        out.append(r)
    return out


def profile_from_record(r: dict[str, Any]) -> SheetProfile:
    return SheetProfile(
        headers=list(r.get("headers") or []),
        header_row=r.get("header_row") if r.get("header_row") not in ("", None) else None,
        row_count=int(r.get("row_count") or 0),
        nonblank_rows=int(r.get("nonblank_rows") or 0),
        max_width=int(r.get("max_width") or 0),
        mean_width=float(r.get("mean_width") or 0.0),
        fill_density=float(r.get("fill_density") or 0.0),
        column_types=list(r.get("column_types") or []),
    )


def train_from_records(records: Iterable[dict[str, Any]], **kw: Any) -> SheetStructureHead:
    recs = list(records)
    X = np.stack([featurize(profile_from_record(r)) for r in recs])
    return fit_sheet_head(X, [r["label"] for r in recs], **kw)


# ── customer-grouped evaluation ─────────────────────────────────────

_SCOPE_ROLE = "scope"


def evaluate_customer_holdout(
    records: list[dict[str, Any]],
    *,
    holdout_customers: Iterable[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    **fit_kw: Any,
) -> dict[str, Any]:
    """Train on some customers, test on customers the head has never seen.

    ``holdout_customers`` given -> one split. Otherwise leave-one-customer-out
    over every customer key. Abstentions count as fallthrough (and as wrong for
    per-class accuracy). ``fallthrough_before`` is the share of test sheets
    that the rules alone fell through on (from each record's ``match``; a
    labelling record with no ``match`` came from the fallthrough census).
    """
    customers = sorted({str(r.get("customer_key") or "") for r in records})
    if holdout_customers is not None:
        folds = [sorted(set(holdout_customers))]
    else:
        folds = [[c] for c in customers]
    per_class_total: Counter[str] = Counter()
    per_class_correct: Counter[str] = Counter()
    n_test = abstain = before = scope_leak = 0
    fold_reports = []
    for held in folds:
        train = [r for r in records if str(r.get("customer_key") or "") not in held]
        test = [r for r in records if str(r.get("customer_key") or "") in held]
        if not test:
            continue
        if len({r["label"] for r in train}) < 2:
            fold_reports.append({"holdout": held, "skipped": "fewer than two labels in train"})
            continue
        head = train_from_records(train, threshold=threshold, **fit_kw)
        f_ab = 0
        for r in test:
            n_test += 1
            per_class_total[r["label"]] += 1
            before += int(r.get("match", "fallthrough") == "fallthrough")
            pred = head.predict_profile(profile_from_record(r))
            if pred is None:
                abstain += 1
                f_ab += 1
                continue
            if pred.label == r["label"]:
                per_class_correct[r["label"]] += 1
            if pred.role == _SCOPE_ROLE and LABEL_TO_ROLE[r["label"]] != _SCOPE_ROLE:
                scope_leak += 1
        fold_reports.append({"holdout": held, "n_train": len(train), "n_test": len(test), "abstained": f_ab})
    n = max(1, n_test)
    return {
        "n_test": n_test,
        "customers": customers,
        "folds": fold_reports,
        "fallthrough_before": round(before / n, 4),
        "fallthrough_after": round(abstain / n, 4),
        "accuracy": round(sum(per_class_correct.values()) / n, 4),
        "per_class_accuracy": {
            c: round(per_class_correct[c] / per_class_total[c], 4)
            for c in LABEL_CLASSES if per_class_total[c]
        },
        "per_class_support": {c: per_class_total[c] for c in LABEL_CLASSES if per_class_total[c]},
        "non_scope_routed_to_scope": scope_leak,
        "threshold": threshold,
    }


# ── serving ─────────────────────────────────────────────────────────

_CACHE: dict[str, Any] = {}


def load_head(path: str | os.PathLike[str] | None = None) -> SheetStructureHead | None:
    """Load the active head, or ``None`` (=> abstain). Never raises."""
    p = Path(path or os.environ.get(HEAD_ENV) or DEFAULT_HEAD_PATH)
    key = str(p)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    hit = _CACHE.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        head = SheetStructureHead.from_json(json.loads(p.read_text()))
    except Exception:
        head = None
    _CACHE[key] = (mtime, head)
    return head


def clear_cache() -> None:
    _CACHE.clear()


__all__ = [
    "LABEL_CLASSES",
    "LABEL_TO_ROLE",
    "SheetProfile",
    "SheetPrediction",
    "SheetStructureHead",
    "profile_sheet",
    "featurize",
    "fit_sheet_head",
    "read_labels",
    "train_from_records",
    "load_head",
    "evaluate_customer_holdout",
    "clear_cache",
]
