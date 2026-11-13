"""Carry saved labels onto PDF form lines the parser now joins (dev only).

``app/parsers/pdf/form_records`` folds a form's CITY / STATE / ZIP into its
address line, and a lone contact's phone into the contact line. The joined
line has new text, so its label key changes and the label saved on the old
address line detaches.

The labeling API already heals a detached label whose text matches a live
line (``healDetachedLabels`` in Platform-infra ``atom-labeling-routes.js``):
it re-keys the row with the page's own key, atom id and compile id, and never
over a label the same person saved on the new line. So this script only
rewrites the saved TEXT of a row on an old head line to the joined text. The
next page load does the rest.

What it does not do:

* Labels on absorbed pieces ("CITY: Phoenix" saved as Drop / fragment) are
  left as they are. They no longer match a line, so they stop showing and stop
  training (``liveLabelKeys``). Nothing is deleted.
* A form-answer copy the parser now drops had the same text as its survivor,
  so the heal re-keys it with no help from here.

Usage (prints counts only; no customer text):

    PG=... python scripts/move_joined_form_labels.py --deal 000037 --pdf <file.pdf> [--filename NAME] [--apply]

Without ``--apply`` it is a dry run. Re-parse the deal on dev first, so the
joined lines exist when the page heals the rows.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t or "").strip().lower())


def joined_heads(pdf: Path) -> list[tuple[str, str]]:
    """``[(old head text, joined text)]`` for one PDF, as the parser joins it now."""
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    out = OrbitBriefPdfParser().parse_artifact_full("move-labels", "a", pdf)
    pairs: list[tuple[str, str]] = []
    for a in out.atoms:
        v = a.value if isinstance(a.value, dict) else {}
        for r in v.get("joined_from") or []:
            if r.get("role") == "head" and _norm(r.get("text")) != _norm(a.raw_text):
                pairs.append((str(r["text"]), str(a.raw_text)))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deal", required=True)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--filename", help="the file's name on the deal, if not the local name")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    pairs = joined_heads(args.pdf)
    fname = args.filename or args.pdf.name
    print(f"deal {args.deal}: {len(pairs)} joined line(s) in this file")
    if not pairs:
        return 0

    import psycopg2

    conn = psycopg2.connect(os.environ["PG"])
    cur = conn.cursor()
    norm_sql = "regexp_replace(lower(trim({col})), '\\s+', ' ', 'g')"
    total_labels = total_links = 0
    for old, new in pairs:
        cur.execute(
            f"SELECT count(*) FROM public.atom_labels WHERE deal_id = %s AND lower(filename) = lower(%s)"
            f" AND {norm_sql.format(col='text')} = %s",
            (args.deal, fname, _norm(old)),
        )
        n_labels = cur.fetchone()[0]
        cur.execute(
            f"SELECT count(*) FROM public.atom_label_links WHERE deal_id = %s"
            f" AND ({norm_sql.format(col='from_text')} = %s OR {norm_sql.format(col='to_text')} = %s)",
            (args.deal, _norm(old), _norm(old)),
        )
        n_links = cur.fetchone()[0]
        total_labels += n_labels
        total_links += n_links
        if args.apply and (n_labels or n_links):
            cur.execute(
                f"UPDATE public.atom_labels SET text = %s WHERE deal_id = %s AND lower(filename) = lower(%s)"
                f" AND {norm_sql.format(col='text')} = %s",
                (new, args.deal, fname, _norm(old)),
            )
            cur.execute(
                f"UPDATE public.atom_label_links SET from_text = %s WHERE deal_id = %s"
                f" AND {norm_sql.format(col='from_text')} = %s",
                (new, args.deal, _norm(old)),
            )
            cur.execute(
                f"UPDATE public.atom_label_links SET to_text = %s WHERE deal_id = %s"
                f" AND {norm_sql.format(col='to_text')} = %s",
                (new, args.deal, _norm(old)),
            )
    if args.apply:
        conn.commit()
    verb = "moved" if args.apply else "would move"
    print(f"{verb}: {total_labels} label row(s), {total_links} link row(s)."
          " Open the deal's labeling page once to re-key them.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
