"""The deal record standard lives in the ``c3_record`` package (repository
root), so the parser's worker, which ships without ``ml/``, builds the same
rows the training export does. Re-exported here for ``ml.c3`` callers."""
from c3_record import *  # noqa: F401,F403
from c3_record import EVENT_KINDS, ROLES, SIDES, build_rows, document_times, speaker  # noqa: F401
