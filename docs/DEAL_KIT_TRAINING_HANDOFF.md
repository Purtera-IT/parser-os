# Deal Kit training — handoff (2026-10-16)

For the session that sets up Deal Kit training on top of the C3 super heads.
Dev only. Prod is untouched until one planned migration.

## The design (the owner's rule)

- **The parser only parses**: files → lines, and values (addresses, amounts,
  dates, ids). It does not decide what a line is.
- **The super heads decide** (the C3 model, run C, served from the Mac over
  Azure Relay): the **base layer** is universal (any field-services company),
  the **Purtera layer** is company policy (keep/reject, reasons, partner
  relationships). Labels raise the ceiling; rules do not.
- **The LLM does small supplementary work only**: filling values on lines a
  head typed, and wording the brief. It never decides types on dev now.
- **Lose nothing**: a head replaces an LLM or rule step only where it is
  measured at least as good against the labels.

## What is live on dev (branch `claude/c3-serve-all-heads`, parser-os)

| Piece | Where | Switch |
|---|---|---|
| Base types every line (Deal Kit / estimating sheets excluded), before typing and again at the end, so no rule overrules it | `app/core/c3_type_base.py` `apply_authority` | `SOWSMITH_C3_AUTHORITY=on`, `SOWSMITH_C3_AUTHORITY_SURE=0.0` |
| Typing: base decides, LLM only fills values | `typed_atom_classifier.py` C3 hook | `SOWSMITH_C3_TYPES=<all 69>`, `SOWSMITH_C3_SURE=0.0` |
| Brief v2 (heads choose, one cited writing call, questions from the catalog) | `app/core/c3_brief.py`, envelope `c3_brief` | `SOWSMITH_C3_BRIEF=on`, `_WRITE=on` |
| Question catalog (154: 149 base, 5 Purtera) + Mac refit | `c3_record/questions.json`, `ml/c3/question_fit.py` | refit at `~/c3-serve/c3_base/question_refit.pt` |
| Head registry (layer, status, what each replaces) | `c3_record/heads.json`, `ml/c3/heads_registry.py` | — |
| Site roster merge / mention filter | `app/core/site_identity.py` | in the envelope build |
| Document-job stage | off: every one of its 145 labeled set-asides was wrong | `SOWSMITH_DOCUMENT_JOB_SCOPE=0` |

Measured on 010043 (validation deal, never trained on; Deal Kit lines
excluded; `_keep` read as "the current type stands"): base line type 67.7% vs
parser/LLM 54.0% overall, 91.8% vs 67.5% when sure (≥0.9).

## What a Deal Kit model gets as input

Per deal, from one dev compile (`deals/<uuid>/parser-os/latest/result.json`
and `orbitbrief/latest/envelope.json` in storage `purpulsedevstg01`, container
`orbitbrief-artifacts`):

- **Atoms**: every line, typed by the base (`atom_type`), with
  `value.c3_label` (`{type, p, act}`; `act: authority` = the base decided) and
  the parser's extracted values. `value.alt_atom_types` keeps what a rule said.
- **Head readings** (in `c3_brief`): per line keep/reject (`read:co_action`,
  Purtera), reason (`co_reason`), SOW section (`sow_section`), noise kind,
  plus `site_line_types`.
- **OrbitBrief context**: `c3_brief.sections` (what goes in the brief, by
  section, each item with `decided_by_layer`), `service`, `questions` (asked /
  already answered with the answering line), `pm_dashboard`, the site roster
  `site_readiness` (`merged_from`, `set_aside_sites`).
- **The kit itself is the target**, never an input. `ml/c3/kit.py` is an
  existing kit model (frozen ModernBERT, reads atoms by section); the owner
  plans to retrain from ~50 correct kits with atoms + OrbitBrief context.

## Rules around the saved Deal Kits (verified 2026-10-16)

- **No compile, envelope or brief step writes a kit.** Kits are written only
  by the app's HTTP routes (`Platform-infra/azure-function-api/shared/pm-dealkit-v3-*`).
- **The one automatic writer is the browser**: with a deal's quoting page
  open, a new OrbitBrief handoff auto-applies a prefill to an UNPUBLISHED kit
  (adds flags / head verdicts; never overwrites or deletes). Published kits
  are locked (409). Publish the kits, or keep their pages closed during compiles.
- Never touch the owner's saved kits. One deal at a time, verified.

## What the labels mean (read before scoring anything)

- `label_type: _keep` = "the current (parser) type is right" — score it as the parser's type.
- Deal Kit and estimating sheets: the base never trains on them; exclude
  their lines from every score (counting them made the base look worse).
- `supplier` is labeled on most lines (us / customer / partner /
  third_party). **"partner" is Purtera-relative** (CDW = partner) but the head
  sits in the base layer, so it answers "third_party" on CDW quote lines.
  Move it to the Purtera layer, or map third_party → partner for known partners.
- `document_job`: 88 of 124 "other job" labels are calls (programme deals), so
  the head over-flags calls; it also called 010043's signed SOW and PO "other
  job" at 1.00 (unexplained).
- Drops (`suppression`, 68,537 reviewed): structural drops are ≥99% right;
  judgment drops (document job, pasted-note dedup, vendor check, section
  heading, meaning dedup, substance gate) are 12–100% wrong.

## Open issues that touch Deal Kit work

- **Quote rows are split into fragments** by the color-driven PDF parser
  (`app/parsers/pdf/_shared.py`): on 010043's `PVDP685.pdf` one line item
  becomes description / part # / item number / price as separate lines.
  BOM lines are exactly what a kit prices, so fix this before training on BOMs.
- The LLM's self-taught "teacher" lessons used to act as corrections; fixed
  (they answer as source `teacher`), but old teacher rows remain in the store.

## How to run things

- Mac base server: LaunchAgent `com.purtera.c3-base` (code `~/c3-serve/code`,
  restart with `launchctl kickstart -k gui/$(id -u)/com.purtera.c3-base`);
  relay LaunchAgent `com.purtera.c3-relay`.
- Deploy dev worker: `gh workflow run deploy-dev.yml -R Purtera-IT/parser-os-worker
  -f parser_os_ref=claude/c3-serve-all-heads -f replaces=claude-c3-serve-all-heads`.
  Never change worker env while a compile runs; changing env right before a
  deploy can put the next compile on the old image.
- Re-parse a deal on dev: POST `purpulse-dev-api-eus2/api/data/admin/compile-queue/enqueue`
  (bearer `ADMIN_FIELD_TECHNICIANS_SECRET`, read from app settings, never printed).
- Training files: `~/c3-local/train/*.json` (17 deals + 010043 validation copy);
  holdout `~/c3-local/holdout/` is spent for run C.
- Mac refits (no GPU): `ml.c3.question_fit`, `ml.c3.judgment_fit`, `ml.c3.head_fit`.
