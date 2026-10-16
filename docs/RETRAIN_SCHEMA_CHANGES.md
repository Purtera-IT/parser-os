# Schema changes for the next retrain

These change the model's schema, so they ride with a retrain (the served
checkpoint checks the schema digest at load).

1. **`col:supplier` → company (Purtera) layer.** "partner" is Purtera-relative
   (CDW, SHI are partners), so a universal head answers "third_party" on lines
   labeled "partner" (010043: partner recall 111/281). Until then
   `c3_record.purtera.supplier_for_purtera` maps third_party → partner for
   the partners in `c3_record/purtera_layer.json`.
2. **Document-job calls.** 88 of 124 other-job labels are calls (programme
   deals); add "this deal" labels on calls from single-job deals before
   retraining, or the head keeps over-flagging calls.
3. **One-sided heads need their other answer labeled** before they are worth a
   loss term: site role (236 job site / 9 vendor), geo mention (27/3), wants,
   intake gap, superseded, small talk.
