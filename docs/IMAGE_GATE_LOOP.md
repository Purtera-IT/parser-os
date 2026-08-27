# The Image Gate Loop

How embedded-PDF-image triage learns. One page; receipts, not marketing.

## Shape

```
channel -> veto -> cards -> taps -> eval-gated retrain -> (better veto) -> ...
```

1. **Channel** — every vlm/store gate verdict logs an attributed silver row
   (`relation='pdf_image_kind'`, id `trn_vlm_<sha16>`, deal_id column,
   pdf/page/region provenance). cpu_gate never logs (self-distillation);
   "no context" never logs (teaches nothing). `pdf_image_vision._log_gate_silver`.
2. **Veto** — a binary skip-vs-meaningful head second-guesses skip verdicts.
   Routing never changes; the veto is a recorded flag.
   * hard band (p >= `SOWSMITH_PDF_IMAGE_VETO_CONF`, 0.88): confident
     disagreement -> `gate_verdict.veto` + `relation='pdf_image_veto'` row,
     provenance `band='hard'`.
   * soft band (`SOWSMITH_PDF_IMAGE_VETO_SOFT_CONF` 0.70 <= p < 0.88): the
     head's uncertain zone -> `gate_verdict.veto_soft` + row `band='soft'`.
     Never a `veto` key, never a PM card — harvest only.
   * either band = disputed image -> crop pixels persisted to blob
     (`orbitbrief-artifacts`, `deals/<deal>/orbitbrief/disputed_crops/<sha16>.png`),
     stamped as `gate_verdict.crop_ref` (or `crop_ref_error` on failure).
3. **Cards** — Orbitbrief-Core builds "Images we skipped — but doubt" PM
   culprit cards from `gate_verdict.veto` ONLY (hard band).
4. **Taps** — PM chips + `tools/build_image_review_queue.py` (hard > soft >
   harvest) feed human grades; `tools/import_image_silver.py` writes them as
   `teacher='silver_audit'` rows. `tools/variety_gap_report.py` says which
   varieties to grade next.
5. **Eval-gated retrain** — deal-held-out eval decides what ships.
   `pdf_image_veto` / `pdf_image_gate_shadow` rows are excluded from the
   neural retrain (`NON_TRAINING_RELATIONS`) — the veto head has its own
   trainer and its own bar.

## The four doctrines (learned 2026-08)

**1. Roles-from-eval.** Ship the direction the model is good at, not the role
you planned for it. The binary head was trained to DEFLECT VLM calls; held-out
eval falsified that ceiling — it is never confident about skips (text features
cannot prove absence: logos and signatures have no text), so deflection = 0%.
The same eval showed 100% accuracy at the 0.88 bar in the meaningful
direction on unseen deals. So the veto shipped and the deflector did not.

**2. Silent-zero-for-infra.** A best-effort path must emit a liveness
receipt: a silent zero and a real zero must never look the same. Receipt: the
training-row blob mirror looked never-fired for weeks — a wrong-container
check plus an env gate that was off on the worker meant mid-day rows silently
vanished, and nothing distinguished "no rows" from "dead mirror" until the
584-row silver set got erased by a stale write-back. Hence: disputed-crop
upload failure stamps `gate_verdict.crop_ref_error` with the exception class
instead of nothing — a dead uploader is observable in every envelope it fails.

**3. Variety-over-volume.** Row counts lie; deal-held-out splits expose
variety shift. Receipt: the chart class — the 8-way kind classifier sat at
44% holdout because one holdout deal owned ALL the charts; no volume of the
train deals' varieties could fix it. The soft-veto band and
`variety_gap_report.py` exist to spend grading time on the varieties the
splits disagree about, not on more rows of what we already have.

**4. Plumbing-over-compute.** Training this head takes 0.8 min on the
M3 Ultra; GPU rental was never needed. The campaign's real costs were channel
rot (a dormant gate with 2 wrong-skip rows and junk teachers; the mirror gap
above) and deploy drift (the core two-branch trap: a deploy from a main that
lacked the neural-heads era regressed prod until the branches were unified).
Budget accordingly: fix the pipes and the deploy line first — the model run
is the cheap part.
