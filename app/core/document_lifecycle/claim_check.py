"""Verify that a receipt supports the CLAIM, not merely that it exists.

The receipt rule caught fabrication but not overreach: the model called 20
documents SOW_SIGNED while quoting ordinary SOW boilerplate ("This Statement of
Work is made by and between..."), which proves the document is a SOW and says
nothing about whether anyone signed it. Nine of those twenty had no signature
language anywhere in the extracted text.

So: types that assert a STATE must show evidence of that state, checked here in
code. A type that cannot prove its state is demoted to the weaker claim, which is
almost always the safe one, and flagged.
"""
import re

EVIDENCE = {
    "SOW_SIGNED": (re.compile(r"(fully executed|executed this|signature|signed by|docusign|"
                              r"/s/|accepted by|authori[sz]ed signature|date signed|countersigned)", re.I),
                   "SOW_DRAFT"),
    "AS_BUILT": (re.compile(r"(as[- ]?built|final configuration|installed configuration)", re.I),
                 "OTHER"),
    "CLOSEOUT_PACKET": (re.compile(r"(close[- ]?out|final acceptance|project complete)", re.I),
                        "OTHER"),
    "CHANGE_ORDER": (re.compile(r"(change order|amendment|revised scope|additional work)", re.I),
                     "OTHER"),
}


def check(doc_type: str, receipt: str, full_text: str):
    """-> (kept_type, demoted_bool, reason). Looks in the receipt first, then the body."""
    rule = EVIDENCE.get((doc_type or "").upper())
    if not rule:
        return doc_type, False, None
    pat, fallback = rule
    if receipt and pat.search(str(receipt)):
        return doc_type, False, None
    if full_text and pat.search(full_text):
        return doc_type, False, "state evidence in body, not in the cited receipt"
    return fallback, True, f"no evidence of the state claimed by {doc_type}"
