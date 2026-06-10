# Gold adjudication guide (1 page) — read before labeling

You're creating the **ground-truth** set that every model is graded against. The teacher
(qwen) answer is deliberately hidden from you — label what's *correct*, not what it guessed.

## Your job
For each row, read `prev` / `text` / `next` and put ONE value in **`gold_facet`**:

`SITE` · `COMMERCIAL` · `WORK` · `COMPLIANCE` · `PARTY` · `TIMING` · `META` · `_keep` · `AMBIGUOUS`

(Optionally put a finer micro-type in `gold_micro`, but `gold_facet` is what matters.)

## The one question (decide by ROLE, not topic)
> Does this text state a discrete thing we're committing to for THIS deal?
> **No → `_keep`** (scaffolding). **Yes → which of the 7 facets.**

| Facet | Use when the text is… |
|---|---|
| **SITE** | a place, site access, or physical-site attribute |
| **COMMERCIAL** | a price, rate, quantity/BOM line, payment term, or total |
| **WORK** | an action/service/task/requirement/deliverable/milestone/acceptance (incl. exclusions, technical steps) |
| **COMPLIANCE** | a rule, certification, approval authority, insurance/bonding, change-order process, contractual obligation |
| **PARTY** | a person/org that **acts** in execution (PM, site contact, approver) |
| **TIMING** | a deadline, blackout window, lead-time/sequencing/dependency |
| **META** | deal-level **identity** metadata: project name, provider name, deal id, **a bare contact on a header** |
| **`_keep`** | scaffolding: navigation, legends, **schema/field definitions** ("Available Fields:"), empty form labels, a sales rep/quote signatory |

## The rulings that resolve the common fights (memorize these)
- **Bare name/contact, no action → `META`.** Acts in the deal (escorts installers, approves) → `PARTY`. Sales rep / quote signatory → `_keep`.
- **Field/schema DEFINITIONS → `_keep`** ("Available Fields: Site Close Date", "ServiceNow Fields:"). A deal **value** in a field ("Project Name: Marriott Tower") → `META`.
- **A quantity of equipment we supply → `COMMERCIAL`** (BOM). A count of existing site inventory → `SITE`.
- **Exclusions / "by others" → `WORK`** (negative scope is still scope).
- **A deadline tied to a milestone → `TIMING`** (the facet is about *when*).

## When to use `AMBIGUOUS` (it's not failure — it's data)
Use it when, after the rulings above, the text genuinely fits **two facets equally** or you
**can't tell the role** even with context. Do NOT use it just because you're unsure of the
micro-type. Rule of thumb: if you'd have to flip a coin between two facets, it's `AMBIGUOUS`.
Expect this to be **~10–18%** of rows — much higher means the rules need work (flag it);
much lower means you're guessing.

## Process (so the gold has a known error rate)
1. **Warm-up:** code the first **30 rows**, then reconcile with the eng owner. Align on
   disagreements *before* doing the rest.
2. **Main pass:** ~1,600 rows. Budget ~3 atoms/min with context = **~2.5–3 days**.
3. **Double-coding:** ~10% of rows are labeled by a second person (or re-labeled by you
   cold) so we can publish a gold self-agreement number — every model score is read
   against it (if gold agreement is 0.90, a model "0.92" is at the gold ceiling).
4. Put anything genuinely confusing in `notes` rather than forcing a label.

## Don'ts
- Don't try to guess the micro-type if the facet is clear — facet is the job.
- Don't label from the deal's topic ("it mentions cameras so SITE") — label by role.
- Don't leave `gold_facet` blank on a row you skipped; mark `AMBIGUOUS` or note why.
