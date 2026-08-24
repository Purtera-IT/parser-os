"""One typing vocabulary, whatever format the sentence arrived in.

Three parsers each carried their own copy of the prose-typing regexes --
markdown, pptx, and universal_parsers -- and all three had diverged: markdown
knew "cancel / hold off / defer", the others did not; universal knew
"customer provides", markdown did not; markdown's constraints knew "access
window / acceptance / closeout", the others knew "lift / compliance /
regulatory". Measured on 800 real held-out sentences: **4.0% received a
different atom TYPE depending on which format they arrived in.**

That is the container deciding the evidence -- the exact rule the routing
audit enforced everywhere else -- surviving one layer deeper, at the typing
seam. This module is the fix in the same shape as every other fix on that
branch: one implementation, shared, so the copies cannot drift again. Each
pattern below is the UNION of the three families, so no format loses recall
it previously had; a sentence can only gain a more specific type, never fall
back to scope_item where a sibling format would have typed it.

The classification order is precedence, not arbitrary: an exclusion sentence
frequently also contains constraint vocabulary ("removal must be completed
after hours"), and the governing fact is the exclusion. Question marks are
strong but "TBD" inside an assumption is still an assumption.

This family is the parse-time COARSE pass. The fine taxonomy (35+ classes)
belongs to the compile-time cascade and its heads; measured on the deal-split
holdout, this regex layer covers 10.3% of fine-typed rows at 33.9% accuracy,
which is why its replacement is the atom run. Until the head is promoted,
this stays -- unified, so at least it is wrong the same way everywhere.
"""

from __future__ import annotations

import re

from app.core.schemas import AtomType

#: Union of markdown / pptx / universal exclusion vocabularies.
EXCLUSION_RE = re.compile(
    r"\b(exclud(?:e|ed|es|ing)|out\s+of\s+scope|not\s+included|not\s+in\s+scope|"
    r"explicitly\s+excludes?|exclusion[s]?:|by\s+others|nic|"
    r"remove\s+from\s+scope|please\s+remove|removed?\s+from\s+the\s+scope|"
    r"cancel(?:led|ling|s)?(?:\s+the)?|cancellation|"
    r"do\s+not\s+include|drop\s+(?:the|from)|deletion?|"
    r"hold\s+off|on\s+hold|defer(?:red)?\s+from|postpone(?:d)?)\b",
    re.IGNORECASE,
)

#: Union of the three assumption vocabularies.
ASSUMPTION_RE = re.compile(
    r"\b(assum(?:e|ed|es|ption|ptions)|we\s+assume|subject\s+to|"
    r"provided\s+by\s+owner|customer\s+provides?|customer\s+supplies)\b",
    re.IGNORECASE,
)

#: Union question family: a trailing question mark, or the open-question
#: vocabulary markdown carried and the others lacked.
QUESTION_RE = re.compile(
    r"\?$"
    r"|\b(tbd|to\s+be\s+confirmed|to\s+be\s+determined|unknown|"
    r"open\s+question|please\s+confirm|need(?:s)?\s+confirmation|"
    r"awaiting\s+confirmation|still\s+(?:tbd|pending|outstanding)|"
    r"to\s+clarify|need(?:s)?\s+clarification|pending\s+(?:answer|response))\b"
    r"|^(?:question|to\s+confirm)\b",
    re.IGNORECASE,
)

#: Union constraint vocabulary.
CONSTRAINT_RE = re.compile(
    r"\b(must|shall|required?|requirement|after[-\s]?hours|escort|badge|lift|"
    r"compliance|regulatory|access\s+window|acceptance|completion|closeout)\b",
    re.IGNORECASE,
)


def classify_prose(text: str) -> AtomType:
    """Coarse type for one prose sentence. Same answer in every format."""
    if EXCLUSION_RE.search(text):
        return AtomType.exclusion
    if ASSUMPTION_RE.search(text):
        return AtomType.assumption
    if QUESTION_RE.search(text):
        return AtomType.open_question
    if CONSTRAINT_RE.search(text):
        return AtomType.constraint
    return AtomType.scope_item
