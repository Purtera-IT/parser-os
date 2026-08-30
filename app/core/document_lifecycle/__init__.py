"""Which lifecycle stage a deal document belongs to, and who may read it.

A deal's artifact folder mixes things that answer different questions. Measured
over 1,254 documents on 305 dev deals (2026-08-29): roughly 60% of them are our
own OUTPUT -- Deal Kits, proposals, SOWs -- being fed back in as evidence to the
heads whose job is to produce exactly those. Only about one document in five is
legitimate quoting evidence.

This package assigns each document a TYPE from a fixed vocabulary and derives its
STAGE and ADMISSIBILITY from that type in code (``taxonomy.py``), so routing is a
table a human owns rather than something a model improvises.

Three findings shaped the design, each of which cost a wrong answer first:

* Filenames lie. Classifying the same corpus from content rather than filename
  changed 32% of the labels, including 46 documents that were unclassifiable by
  name alone and turned out to be scope drafts.
* A receipt proves the quote exists, not that it supports the claim. The model
  called 20 documents SOW_SIGNED while quoting ordinary SOW boilerplate; nine had
  no signature language anywhere. ``claim_check.py`` re-tests state claims.
* HubSpot strips document attachments from the mirrored .eml -- verified on the
  twelve largest emails in the corpus, none carried one -- but the CRM still
  records the association in ``hs_attachment_ids``. That is how a document is
  linked back to the message that delivered it, which is what separates "sent for
  review" from "signed".
"""
from .taxonomy import TAXONOMY, TYPES, normalise, route
from .claim_check import check as check_state_claim

__all__ = ["TAXONOMY", "TYPES", "normalise", "route", "check_state_claim"]
