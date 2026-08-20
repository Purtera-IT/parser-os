"""Coverage: what the parsers did NOT read.

These modules were under ``app/parsers/`` and are not parsers. A parser turns
bytes into atoms; these inventory every region of a file straight from the
bytes -- deliberately independent of the production parser -- and reconcile
that inventory against what the parser emitted, so every region is either
COVERED or explicitly MARKED.

That is the coverage census, and the distinction is the point: it is an
INVARIANT, not a prediction. It never learns and never abstains, because its
value is being able to say "this document had 41 regions and you read 38" with
certainty. A parser that scores itself is worthless; a census that is
independent of the parser is the only thing that can tell you what was missed.

Filed apart from the readers for the same reason it is written apart from them.
"""

from app.coverage.census import census, reconciled_census

__all__ = ["census", "reconciled_census"]
