"""A key for a human label that survives a re-parse.

Atom ids are not stable enough to hang a label on: some are hashed from
their own ``atom_type`` (table_rollup.py), so re-typing an atom -- the very
thing a label does -- changes its id, and the old labelers keyed atoms by
list position. A label is instead keyed by where the text sits and what it
says: deal, source filename, page, normalized text. A re-parse that keeps the
same line on the same page re-attaches to the same label.

The purpulse API computes the same key in JavaScript
(Platform-infra ``shared/atom-labeling.js`` ``labelKey``); the shared vector
in ``tests/test_label_key.py`` pins both to one answer.
"""
from __future__ import annotations

import hashlib
import re


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def label_key(deal_id: str, filename: str, page: int | str | None, text: str) -> str:
    page_s = "" if page is None else str(page)
    raw = "|".join([str(deal_id or "").strip(), _norm(filename), page_s, _norm(text)])
    return "lbl_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
