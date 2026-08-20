"""Interpretation: structure in, meaning out. This is where heads live.

The decode package answers what a document SAYS. This one answers what it
MEANS, and the split matters because the two have opposite economics:

    decoding        one correct answer, no judgment, buy the best library
    interpretation  a judgment, correctable by a PM, and therefore learnable

Everything here is a readout in waiting. A readout has three properties the
scattered heuristics it replaces do not:

    one entry point   so a correction has somewhere to land
    a confidence      so it can abstain instead of guessing
    a recorded reason so a PM's disagreement is attached to something specific

The reason to consolidate rather than tidy in place: the architecture calls for
one shared encoder trained on every correction from every task, POOLED, because
ten tasks with a few hundred labels each are ten badly-fit models while the same
labels in one representation are a few thousand examples. Pooling is impossible
while one judgment is implemented three times behind three signatures -- there
is no single thing to correct, and a fix to the PDF path teaches the spreadsheet
path nothing.
"""

from app.interpret.table_kind import TableJudgment, judge_table

__all__ = ["TableJudgment", "judge_table"]
