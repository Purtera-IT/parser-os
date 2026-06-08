# Architecture proposal — the trifecta gate/typer (FAST · LEARNS-FROM-TEXT · ACCURATE)

State: 2026-06-08. This is the target architecture for the atom keep-vs-typed gate and
fine-typing, designed from what we *measured* this session (see BRAINDUMP_SESSION_2026-06-08.md).

## The goal (the trifecta)
1. **FAST** — millisecond inference on the majority of atoms.
2. **LEARNS THROUGH TEXT** — a PM correction improves it *instantly*, no retraining.
3. **ACCURATE** — high precision, guess-free (abstain rather than emit a wrong label).

A *single model* cannot be all three for a reasoning-dependent boundary (a fast/instant-learning
method caps at embedding separability ~0.82; a reasoning model is slow). **So the trifecta is
solved at the SYSTEM level by a confidence-routed cascade.** Critically, we make the fast tier as
strong as possible by fixing the *representation*, not by adding a classifier head.

## Why the naive attempts capped (measured this session)
- Frozen-embedding classifier / kNN: ~0.65 (space organized for general meaning, not for keep-vs-typed).
- LoRA classifier head on 8B embedding, clean labels: **~0.79–0.82** — a classifier head *pattern-matches*; it can't replicate the multi-step *reasoning* the labeler uses.
- More data HURT (noisy auto-labels); cleaner labels barely moved it (~0.77→0.79). So the cap is the
  **model class + the irreducible ambiguity** (qwen3:32b vs 14b only agree ~85% on the boundary), NOT the label volume.
- kNN failed for the same reason the classifier did: **the neighbors aren't sorted by the decision** —
  same-meaning atoms have opposite labels, so the neighborhood is mixed.

## The fix: a 4-layer cascade with a metric-learned fast tier

### Layer 1 — Representation: SUPERVISED-CONTRASTIVE encoder (the key change)
Fine-tune the encoder with a **supervised contrastive / metric loss** on the CLEAN rubric labels:
pull same-label atoms together, push opposite-label atoms apart. Result: a space **organized around
the keep-vs-typed (and facet) boundary**, where keep and typed physically separate.
- This is the difference from what we tried — we trained a *classifier head* (0.82); instead we
  reshape the *space* so the geometry encodes the decision.
- Refit periodically (eval-gated) on the grown clean log. Monotonic — never promote a regression.

### Layer 2 — Fast tier: kNN over the contrastive space + the feedback store
- Embed atom → k nearest neighbors in the labeled store → distance-weighted vote = label; agreement
  = confidence.
- **LEARNS THROUGH TEXT INSTANTLY**: every PM correction / new labeled atom is appended to the store
  and is usable on the very next atom — zero retraining. This is the property an LLM/classifier lacks.
- Inference is an embed + ANN lookup = milliseconds.

### Layer 3 — Confidence gate (guess-free routing)
- kNN confidence ≥ τ (neighbors agree AND close) → EMIT the label, deflect the LLM. High precision.
- else → ABSTAIN → route to Layer 4. (τ tuned for a precision floor, e.g. ≥0.95 on the emitted slice.)

### Layer 4 — Reasoning fallback: the LLM (or a small distilled instruct model)
- Only the genuinely-ambiguous minority reaches here.
- Its decisions are LOGGED → appended to the store → **the fast tier's coverage grows over time**
  (more of the space becomes confidently handled, fewer LLM calls each week).

## The self-improvement flywheel
- PM corrects an atom → store (instant) → next similar atom right.
- LLM fallback decides an atom → store → fast tier covers more next time.
- Periodically: re-run the contrastive fit on the grown clean log → space sharpens → kNN improves.
- All eval-gated + rollback: monotonic, guess-free, universal (rubric-defined labels, no per-deal hacks).

## Why this hits the trifecta where one model can't
| Property | How |
|---|---|
| FAST | kNN handles the confident majority in ms; LLM only on the hard minority. |
| LEARNS-FROM-TEXT | kNN store: new examples used instantly, no retrain. |
| ACCURATE | contrastive space makes the fast tier strong; reasoning fallback covers the ambiguous; abstain = no wrong emits. |

## Honest limits (do not oversell)
- The boundary is **partly irreducible**: even two strong reasoning models agree only ~85%. No space
  separates a question without a single answer. The cascade *routes* that ambiguity to reasoning — it
  doesn't pretend to resolve it. Expect the fast tier to clear the 0.82 wall on the *separable* cases
  and grow coverage over time, with the ambiguous remainder always abstaining.
- "Perfect" = the *system* is fast + self-improving + never emits a guess. Not a single model at 1.0.

## Fine-typing (within "typed") uses the same machine
Same contrastive space + kNN, but multi-class over the 7 universal FACETS (SITE/COMMERCIAL/WORK/
COMPLIANCE/PARTY/TIMING/META — = the dashboard sections; see runpod_detector/RUBRIC.md). Collapse the
41 micro-types to facets (measured 0.846 two-model agreement vs un-learnable 41-way). Rubric the 7
fuzzy types. Same cascade + flywheel.

---

# Parallel system — the schematic MULTI-CLASS detector (vision)
For the OrbitBrief schematic understanding (Luke & Danny's labels), the target is detection +
association, not a single model:
1. **Multi-class detector (YOLO):** classes = device, device_tag, room, room_tag, note_block, legend,
   arrow, callout. (Rooms better as segmentation polygons.)
2. **OCR** reads the text regions (device tags, room numbers).
3. **Geometric association (the "insanely powerful" layer):** device_tag→device (nearest/above),
   device→room (inside polygon), arrow→target (endpoints), room_tag/callout→note (number lookup).
   Detection finds *where*; association produces *meaning* ("CAM-3 = PTZ camera in Room 214, per Note 5").
4. **Self-supervised pretrain** the backbone on all unlabeled schematic pages (corpus has 1,350) →
   the few labels go further; **active-learning loop**: detector pre-labels new pages → human verifies.
5. **Data reality (measured intuition):** big/consistent classes (note_block, legend, room) need only
   ~30–50 diverse examples; tiny/variable classes (device, device_tag) need volume → more pages.
6. **Per-firm symbol meaning via the LEGEND** (the Rosetta stone) — never memorize symbols across firms;
   detect symbol + read that firm's legend + map to a UNIVERSAL device type.

The detector is **data-bound** (measured held-out-firm mAP@50 = 0.11 on 24 firms) — the lever is more
labeled firms, which the Luke/Danny pipeline + corpus + scraper provide.
