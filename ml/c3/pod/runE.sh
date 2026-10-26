#!/usr/bin/env bash
# Run E: the parts run C switched off, turned back on, from the live checkpoint.
# Run C froze the encoder, skipped the teacher and trained lines and open_links
# only. Here the top 2 encoder blocks learn (run B's setting, at 2e-5), links
# and cards train. The teach stage is off by default (TEACH=0): run A's teach
# and run D's teach@1 both broke jdg:task_tier (1.000 to .656 / .545), and the
# WHY carry terms are teacher-free. After the loop a pick step (ml/c3/pick.py)
# keeps the checkpoint with the best macro over the gated heads whose every
# head holds the live model's own score on 010043 (FLOORS), not the run's
# start check (run C read under the new texts and tags).
#
# Every loss weight is set here (the loss table below; the reasoning and the
# ablation plan are in runE/loss_plan.md). The new terms (why_contrast,
# why_alone, keyword_dropout, layout_lock, link_why_contrast, link_why_alone,
# and numbers when its stage opens) grow from 0 over RAMP passes (2) from the
# first step each is on. The link_why terms (ml/c3/link_why.py) train the
# links' notes into the relation head; like relations they open at links, add
# no forward pass, and so leave the graph count (and ATOMS) as it is.
#
# The layout meaning lock is on: the run E files carry layout_rewrites on table
# rows, and losses.layout_lock_loss (runE-data, 2eaa5ef) reads each part once
# more with one rewrite per row and holds the saved answers and the real-layout
# reading. No teacher needed. It is one more forward and backward on every part
# with rewrites (most of them). keyword_dropout is one more full forward with
# grads, so with both on a step holds three graphs: ATOMS defaults divide by
# the graph count (the lock reuses the step's encoded descriptions).
# entity_swap (losses.entity_swap_loss, ml/c3/entity_swap.py) is the meaning
# lock for names, which pilot C never trained and gate 2 failed on: the part
# read once more with its people, companies, places and IDs swapped
# consistently (gate 2's entity edits, training's own stand-ins), holding the
# saved answers and the first reading. One more forward on every part with a
# name in it, so it counts as a graph too (ENTITY_SWAP=0 turns it off).
# meaning_flip (losses.meaning_flip_loss, ml/c3/meaning_flip.py) is the other
# half: the contrast test (runE/contrast_gate.py) showed pilot C stays put when
# the one meaning-bearing word changes (supplier party 1/13 and 0/9, an explicit
# exclusion 3%, a street added to a city/zip row 0/46). The part is read once
# more with some labeled lines given a minimal edit whose new answer is certain
# from the head definitions (party swapped, street added, explicit leave-out,
# fact <-> small talk; templates disjoint from the gate's), and each edited line
# trains toward its NEW answer (cross-entropy, from the saved gold). One more
# forward, so it counts as a graph (MEANING_FLIP=0 turns it off).
# The teacher's own rewrites
# (supercharge.teach_rewrites) stay unwired in loop.py. FLIPS=1 adds the flips
# lesson (teach_flip at 0.2).
#
#   TRAIN_DIR   training files (default "$C3/runE/train_presow $C3/runE/train_plan"); several dirs, space
#               separated, train together (a quoting run: "$C3/runE/train_presow
#               $C3/runE/train_estimator"). Every NNNNNN.json or NNNNNN@<view>.json
#               there trains, minus the validation deal and the fixed holdouts,
#               unless TRAIN_DEALS lists the file names (without .json). A listed
#               validation or holdout deal is refused, and every chosen file's
#               own deal_id is checked again before the run
#   VALID_FILE  the validation files, space separated. Default: 010043's quoting
#               view (view pre_quote) AND its planning view, found by the view
#               field in the TRAIN_DIRs and $C3/runE/train_presow / train_plan;
#               both are required. Never trained, in any view
#   INIT        start checkpoint (default $C3/runC.pt.open_links.slim, the live
#               one; its md5 is checked unless INIT_SHA256 pins a refitted checkpoint)
#   INIT_SHA256 exact SHA256 for a refitted INIT; checked before any training.
#   LAYOUT_LOCK KEYWORD_DROPOUT ENTITY_SWAP MEANING_FLIP WHY_CONTRAST WHY_ALONE WHY_SWAP TYPE_WEIGHT RAMP
#   WHY_ALIGN WHY_SUFFICIENCY LINK_WHY_CONTRAST LINK_WHY_ALONE  override the
#               loss table (0.5 0.3 0.3 0.3 0.3 0.2 0.1 5 2 0 0 0.3 0.2); 0 turns a term off. The ablation configs are in loss_plan.md
#   ATOMS       lines per part. Default from the card and the graphs a step
#               holds (1 + layout_lock on + keyword_dropout on + entity_swap on + meaning_flip on):
#               8192 / graphs at >=120 GB (H200), 4096 / graphs at >=70 GB, rounded down to 512.
#               With all four on: 1536 / 512 (three of them: 2048 / 1024). Under 70 GB it refuses. Prove the
#               peak with SMOKE=1 C3_SMOKE_LARGEST=1 and set ATOMS from it
#   FLOORS      the live model's per-head scores on 010043 ($C3/runE/live_floors.json)
#   KEEP        checkpoints kept on disk, plus the best (3)
#   RESUME=1    carry on from $CK.resume.pt after a pod failure (same command
#               otherwise; see the restart line the launch prints)
#   TINY=1      the CPU stand-in (loop --tiny): no INIT, no GPU, ATOMS 512
#   VIEW_TAG=1  train the quoting/planning input tag (default on); rebuilt per-line
#               views are required. 0 retains the untagged input contract.
#   FG=1        run in the foreground (tests), not detached
#   LINES_GATE  the lines@2 valid_exact stop (0.50); 0 turns it off
#   LINES LINKS CARDS NUMBERS OPEN_LINKS   passes per student stage (12 6 6 2 4);
#               numbers trains the hours value and the hours per / qty heads
#   ACCEPT_DRAFTS the owner's approval manifest ({"files": [{"file", "sha256"}]}):
#               a file of assistant drafts only (runE/train_estimator) trains
#               only when listed here; otherwise it is left out
#   INIT_SCHEMA_REF  the commit INIT was trained on, when INIT predates shape
#               digests (loop.py --init-schema-ref). Default: INIT_SHAPE, the saved
#               run C schema (ref 0b6efce). The init allows text changes and the
#               heads added since (NEW_HEADS)
#   NEW_HEADS   heads INIT does not have (loop.py --allow-new-heads). Default: derived
#               from the schema diff (ml/c3/new_heads.py, INIT_SCHEMA_REF or INIT_SHAPE);
#               set by hand it must equal the derived list
#   TEACH=0     (default) no bind/teach and the 7B is never loaded; TEACH=1 turns
#               the gated teach stage back on
#   TEACHER     saved teacher (default $C3/runA.pt.teacher)
#   TEACHER_CONTINUE=1  one more teacher pass first (+~2.6 h on an H200)
#   TEACH_STEPS VIEW_PARTS LESSON_GOLD   teach passes (2), viewed parts (24),
#               gold steps per lesson step on unviewed parts (2), as run D
#   TIER_GATE   the in-loop stop for teach/flips: valid_jdg_task_tier floor (0.98)
#   PICK_Z PICK_GAP  the pick's noise allowance (2 standard errors) and the
#               largest growth of fit_exact - valid_exact over the start's (0.05)
#   CHECK_EVERY checks (and a saved copy, ~3.4 GB) every N passes (2)
#   SMOKE=1     every stage the run trains, once on two parts, every term at its
#               full weight (no ramp), no gates, peak memory printed (torch and
#               nvidia-smi); with C3_SMOKE_LARGEST=1 the largest part
#   DESCRIBED_ALONE CARD_ALONE CARD_CONTRAST CO_WHY_CONTRAST CO_WHY_ALONE NUMBER_WHY_ALONE
#   JUDGMENT_WHY_ALONE COMPANY_R_GRAD  the why2 terms (0.3 0.2 0.2 0.2 0.2 0.1 0.2; company
#               gradient into the trunk 0.1); FREE_R_ONLY=0 keeps the critical heads' free
#               rows reading h (ml/c3/why2.py, free_rows.py)
#   COMPANY_REFIT=1 (default) the pick also runs the pretend-new-company check
#               ($CK.company_refit.json): the base must stay bit-identical
#   C3_WHY_BANK the serve-time WHY bank (ml/c3/why_bank.py): unset = off (default)
#   SIGNAL_AUDIT (1 with SMOKE=1; 0 skips) the pre-launch gates, numbers in
#               $C3/out/smoke_gate.md. Before the loop, ml/c3/signal_audit.py on
#               the training files (the run's model, loss table and stages; one
#               step): every labeled signal (label fields, readings, note kinds,
#               link relations and notes, card heads, notes and fields, deal
#               answers) is removed in turn and the step's gradient compared. NO-GO
#               if a signal with >= SIGNAL_MIN (50) rows is UNUSED or ZERO-GRAD and
#               not in SIGNAL_ALLOWLIST (ml/c3/runE/signal_allowlist.json, with a
#               reason), if a reasoning-text kind with >= SIGNAL_KIND_WORDS (10000)
#               words reaches no loss, or if under SIGNAL_COVERAGE (0.95) of all
#               reasoning words do. After the loop, ml/c3/smoke_checks.py on the
#               smoke's checkpoint: NO-GO if a critical head's h-path share is
#               above MAX_H_SHARE (0.10), or if one company-layer refit step moves
#               any base output
#   PICK_ONLY=1 only the pick step, on an existing $C3/out/runE.pt.checks.json
#   GRAPH       cross-line flags (ml/c3/graph.py, arch_graph_review fixes 2-7): default rune =
#               the view mask (a later line can change an earlier reading; the cut is the
#               leak boundary) with cut dropout 0.3, the hop both ways with per-relation
#               maps, governs child->parent with a no-parent column, a second hop, the hop
#               into h, the pointer edge, no near_miss message, table/document nodes with
#               pointers, card slots. off = run C's graph. Saved in every checkpoint, so
#               serve builds the same graph
#   VALID_ATOMS the validation window (8192, serve's): the checks and the pick read what
#               serve reads; the training window is recorded in each checkpoint
#   DEAL        passes of the deal stage (2), at a rate floor of DEAL_RATE (0.3) x LR
#   REFIT=1     (default) after the pick, refit the Questions overlay on the picked
#               checkpoint ($CK.best.question_refit.pt, base md5 inside); 0 = serve without it
#
# Pilot A's three measured causes (~/c3-local/runE/pilot/rootcause/rootcause.md), each fixed
# behind a flag that is on here:
#   LATTICE_DOC=1  (M1) the asked lattice's slots read their own document only (graph flag
#               lattice_doc, saved in the checkpoint, so serve masks the same way); other
#               documents reach a line through the deal graph, the link hop, the table /
#               document nodes and the pointers. 0 = the old cross-document slots
#   TRUNK_LR_SCALE TRUNK_PULL ANCHOR_KL ANCHOR_HEADS  (M1, the drift) the shared trunk
#               (deal graph blocks, lift, ln) at LR x 0.1, pulled 1% of its drift per step
#               back to INIT (decoupled L2), and KL to INIT's predictions on the unlabeled
#               rows of the anchor heads at 0.5 (ml/c3/anchor.py; loss_plan.md "Trunk
#               anchor"). 1 / 0 / 0 turn each off
#   EMBED_CONTRAST=0 (M2) the line types' "Unlike X ..." clauses stay out of the embedded
#               answer texts (schema.EMBED_CONTRAST; the registry and the WHYs keep them);
#               1 = b395fbb's texts
#   TEXT_PINS   (M2) the heads the description gate failed read INIT's texts
#               (ml/c3/runE/text_pins.json over runE/init_texts.json; empty = that file,
#               0 = every head's new text, or a path to another pin file)
#   DESC_GATE=1 (M2) before the loop, INIT is read under INIT's texts and this code's
#               (ml/c3/desc_gate.py) on the validation files plus DESC_GATE_DEALS (space
#               separated, labeled; DESC_GATE_DRAFTS read drafts as gold): NO-GO when the
#               text change alone costs a head with >= 20 rows more than 2 points. Refit
#               the affected heads (ml/c3/head_fit.py --init-shape ... --keys ...) before launching,
#               never past it; then INIT=<refit> INIT_SHA256=<verified hash>
#               DESC_OLD_CK=<the live checkpoint>
source "$(dirname "$0")/common.sh"
# Pilot A M2: the embedded type texts, the same in every python below and in serve (the
# schema digest records it, so a checkpoint never loads under the other texts).
export C3_EMBED_CONTRAST=${EMBED_CONTRAST:-0}
export C3_TEXT_PINS=${TEXT_PINS:-}
cd "$REPO"
CK=${CK:-$C3/out/runE.pt}
INIT=${INIT:-$C3/runC.pt.open_links.slim}
LIVE_MD5=5f49c472d5cbd1ff937cad2e7764c0a5   # c3_base/MANIFEST.json: runC.pt.open_links.slim
TEACHER=${TEACHER:-$C3/runA.pt.teacher}
TEACH=${TEACH:-0}
TINY=${TINY:-0}
FLOORS=${FLOORS:-$C3/runE/live_floors.json}
KEEP=${KEEP:-3}
PICK_Z=${PICK_Z:-2} PICK_GAP=${PICK_GAP:-0.05}
[[ -f "$FLOORS" ]] || { echo "no FLOORS $FLOORS (the live model's per-head scores on 010043)" >&2; exit 1; }

# The best checkpoint by the macro over the gated heads, every head at or above
# the live model's own score (less noise); the start (the live model) is the
# fallback. Writes <ck>.pick.json and links <ck>.best.
pick() {
  python -m ml.c3.pick --checkpoint "$1" --init "$INIT" --floors "$FLOORS" --z "$PICK_Z" --gap "$PICK_GAP" || return
  # Pretend a new company (ml/c3/company_refit.py): refit only the company layer
  # on the training deals but 3, base frozen; the base's outputs on the validation
  # deal must stay bit-identical. COMPANY_REFIT=0 skips it.
  # The picked file; a tiny run's fallback (an INIT that is not there) reads the last one.
  local refit_ck; refit_ck=$(readlink "$1.best"); [[ -f "$refit_ck" ]] || refit_ck=$(readlink -f "$1")
  if [[ "${COMPANY_REFIT:-1}" == 1 && ${#train_files[@]} -gt 0 && -f "$refit_ck" ]]; then
    python -m ml.c3.company_refit --checkpoint "$refit_ck" --train "${train_files[@]}" \
      --probe "${valid_files[@]}" --max-atoms "$ATOMS" --out "$1.company_refit.json"
  fi
}
if [[ "${PICK_ONLY:-0}" == 1 ]]; then pick "$CK"; exit; fi

# Training and validation files. A file is NNNNNN.json or a view of the deal,
# NNNNNN@<view>.json; the deal number before the @ decides what may train.
TRAIN_DIR=${TRAIN_DIR:-$C3/runE/train_presow $C3/runE/train_plan}
read -ra dirs <<< "$TRAIN_DIR"
for d in "${dirs[@]}"; do [[ -d "$d" ]] || { echo "no TRAIN_DIR $d" >&2; exit 1; }; done
FIXED=$(python -c "from ml.c3.holdout import FIXED_HOLDOUT; print(' '.join(sorted(FIXED_HOLDOUT)))")
NEVER=" $VALIDATION $FIXED "
train_files=()
if [[ -n "${TRAIN_DEALS:-}" ]]; then
  for name in $TRAIN_DEALS; do
    [[ "$NEVER" == *" ${name%%@*} "* ]] && { echo "TRAIN_DEALS lists $name: validation or holdout, never trained" >&2; exit 1; }
    hit=""
    for d in "${dirs[@]}"; do [[ -f "$d/$name.json" ]] && { train_files+=("$d/$name.json"); hit=1; }; done
    [[ -n "$hit" ]] || { echo "missing $name.json in $TRAIN_DIR" >&2; exit 1; }
  done
else
  for d in "${dirs[@]}"; do
    for f in "$d"/*.json; do
      name=$(basename "$f" .json)
      [[ "$name" =~ ^[0-9]{6}(@[a-z_]+)?$ ]] || continue
      [[ "$NEVER" == *" ${name%%@*} "* ]] && continue
      train_files+=("$f")
    done
  done
fi
(( ${#train_files[@]} )) || { echo "no training deals in $TRAIN_DIR" >&2; exit 1; }
# The file name can lie; the deal_id inside is what the loop trains. A file
# whose every row is an assistant draft trains only when the owner approved it
# (ACCEPT_DRAFTS, a manifest of file names and sha256s); otherwise it is left
# out here, so it costs no step and teaches nothing.
kept=$(python - "$NEVER" "${ACCEPT_DRAFTS:-}" "${train_files[@]}" <<'PY'
import hashlib, json, os, sys
from ml.c3.data import _is_a_person
never = set(sys.argv[1].split())
approved = {}
if sys.argv[2]:
    approved = {os.path.basename(r["file"]): r["sha256"] for r in json.load(open(sys.argv[2]))["files"]}
seen = {}
for path in sys.argv[3:]:
    raw = open(path, "rb").read()
    blob = json.loads(raw)
    deal_id = str(blob.get("deal_id") or "")
    base = deal_id.partition("#c3part")[0].partition("@")[0]
    if base in never:
        sys.exit(f"{path}: deal_id {deal_id} is the validation deal or a holdout, never trained")
    if deal_id in seen:
        sys.exit(f"{path}: deal_id {deal_id} also in {seen[deal_id]}")
    seen[deal_id] = path
    if blob.get("skip") or not blob.get("atoms"):
        # An empty view (no input lines: 000002's pre-quote view) is an empty part the loop cannot train.
        print(f"skip {os.path.basename(path)}: {blob.get('skip_reason') or 'no atoms'}", file=sys.stderr)
        continue
    rows = [r for k in ("labels", "judgments", "deal_answers") for r in blob.get(k) or [] if isinstance(r, dict)]
    if rows and not any(_is_a_person(r.get("labeler")) for r in rows):
        name = os.path.basename(path)
        if name not in approved:
            print(f"skip {name}: assistant drafts only, not on ACCEPT_DRAFTS", file=sys.stderr)
            continue
        if hashlib.sha256(raw).hexdigest() != approved[name]:
            sys.exit(f"{path}: changed since the owner approved it")
    print(path)
PY
) || exit 1
train_files=()
while IFS= read -r f; do [[ -n "$f" ]] && train_files+=("$f"); done <<< "$kept"
(( ${#train_files[@]} )) || { echo "no training deals left in $TRAIN_DIR" >&2; exit 1; }
# Check the selected inputs before any model/GPU work. The loop rechecks the
# metadata carried by its actual loaded parts (also on direct calls/resume).
balance_args=(--deal-cap "${DEAL_CAP:-0.15}" --head-cap "read:feeds_decision=${HEAD_CAP:-0.15}")
python -m ml.c3.balance --train "${train_files[@]}" "${balance_args[@]}"

# Validation: 010043's quoting view and its planning view, both (the quoting
# view alone is 494 typed lines and 3 task-tier cards: one card is 33 points).
if [[ -z "${VALID_FILE:-}" ]]; then
  VALID_FILE=$(python - "$VALIDATION" "${dirs[@]}" "$C3/runE/train_presow" "$C3/runE/train_plan" <<'PY'
import glob, json, os, sys
deal, found = sys.argv[1], {}
for d in sys.argv[2:]:
    for p in sorted(glob.glob(f"{d}/{deal}.json") + glob.glob(f"{d}/{deal}@*.json")):
        view = json.load(open(p)).get("view") or ""
        found.setdefault(view, os.path.realpath(p))
missing = [v for v in ("pre_quote", "planning") if v not in found]
if missing:
    sys.exit(f"validation: no {deal} view {', '.join(missing)} in {' '.join(sys.argv[2:])}")
print(found["pre_quote"], found["planning"])
PY
) || exit 1
fi
read -ra valid_files <<< "$VALID_FILE"
for f in "${valid_files[@]}"; do [[ -f "$f" ]] || { echo "no VALID_FILE $f" >&2; exit 1; }; done
echo "train ${#train_files[@]} files from $TRAIN_DIR: $(for f in "${train_files[@]}"; do basename "$f" .json; done | tr '\n' ' ')"
echo "validation ${valid_files[*]}"

# The start checkpoint is the live one.
if [[ "$TINY" != 1 ]]; then
  [[ -f "$INIT" ]] || { echo "no INIT $INIT" >&2; exit 1; }
  if [[ -n "${INIT_SHA256:-}" ]]; then
    got=$(python - "$INIT" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
        h.update(block)
print(h.hexdigest())
PY
    )
    [[ "$got" == "$INIT_SHA256" ]] || { echo "INIT sha256 mismatch" >&2; exit 1; }
  elif [[ "${INIT_MD5:-}" != skip ]]; then
    if command -v md5sum >/dev/null; then got=$(md5sum "$INIT" | cut -d' ' -f1); else got=$(md5 -q "$INIT"); fi
    [[ "$got" == "$LIVE_MD5" ]] || { echo "INIT md5 $got is not the live checkpoint ($LIVE_MD5); INIT_MD5=skip to start elsewhere" >&2; exit 1; }
  fi
fi

# The loss table (runE/loss_plan.md). Every LossWeights term is named, so a
# default changing in losses.py cannot change this run. A stage still trains
# only the terms its gold_weights turns on (relations from links, judgments from
# cards, numbers/literals from numbers, deal_numbers from deal/open_links).
LAYOUT_LOCK=${LAYOUT_LOCK:-0.5} KEYWORD_DROPOUT=${KEYWORD_DROPOUT:-0.3}
ENTITY_SWAP=${ENTITY_SWAP:-0.3}   # names/companies/places/IDs swapped: same answers (gate 2 entities)
MEANING_FLIP=${MEANING_FLIP:-0.3}   # one meaning word edited: the certain NEW answer (contrast gate's logic)
WHY_CONTRAST=${WHY_CONTRAST:-0.3} WHY_ALONE=${WHY_ALONE:-0.2} WHY_SWAP=${WHY_SWAP:-0.1}
TYPE_WEIGHT=${TYPE_WEIGHT:-5} RAMP=${RAMP:-2}   # 5: type was ~43% of head CE at 10
# HEAD_WEIGHTS: more per-head loss weights, space separated "key=w" (e.g. "col:admission=5 jdg:facility=4");
# pilot C weighted only line type, and admission / the site judgments lagged live (2026-10-26).
head_weight_args=(--head-weight col:label_type="$TYPE_WEIGHT")
for hw in ${HEAD_WEIGHTS:-}; do head_weight_args+=(--head-weight "$hw"); done
WHY_ALIGN=${WHY_ALIGN:-0} WHY_SUFFICIENCY=${WHY_SUFFICIENCY:-0}   # E-base ablation: 0.1 / 0.5 (run C)
LINK_WHY_CONTRAST=${LINK_WHY_CONTRAST:-0.3} LINK_WHY_ALONE=${LINK_WHY_ALONE:-0.2}   # link notes (link_why.py)
GRAPH=${GRAPH:-rune}
# Pilot A M1: own-document lattice slots (graph.py lattice_doc).
LATTICE_DOC=${LATTICE_DOC:-1}
[[ "$GRAPH" != off && "$LATTICE_DOC" == 1 && "$GRAPH" != *lattice_doc=* ]] && GRAPH="$GRAPH,lattice_doc=1"
# Pilot A M1 (the drift): the trunk slower than the heads, held near INIT (ml/c3/anchor.py,
# runE/loss_plan.md "Trunk anchor"). The KL heads must all be INIT's (no new head).
TRUNK_LR_SCALE=${TRUNK_LR_SCALE:-0.1} TRUNK_PULL=${TRUNK_PULL:-0.01} ANCHOR_KL=${ANCHOR_KL:-0.5}
ANCHOR_HEADS=${ANCHOR_HEADS:-col:label_type,col:admission,col:supplier,col:about,read:site}
CITATIONS=${CITATIONS:-0.1} DERIVED_CARDS=${DERIVED_CARDS:-0.5} TABLE_KIND=${TABLE_KIND:-0.2}
COHERENCE=${COHERENCE:-0.2} NODE_POINTERS=${NODE_POINTERS:-0.2}   # graph_losses.py
weights=(
  heads=1 conduct=1 numbers=0.2 relations=0.5 governs=0.5
  judgments=1 judgment_rationale=0.2 literals=0.4 deal_numbers=0.2
  why_align="$WHY_ALIGN" why_sufficiency="$WHY_SUFFICIENCY"   # 0: replaced by why_contrast / why_alone
  why_contrast="$WHY_CONTRAST" why_hard_negative=4
  why_alone="$WHY_ALONE"
  why_swap="$WHY_SWAP" why_swap_cap=64
  keyword_dropout="$KEYWORD_DROPOUT" keyword_dropout_p=0.3 keyword_dropout_cap=256
  layout_lock="$LAYOUT_LOCK"
  entity_swap="$ENTITY_SWAP"
  meaning_flip="$MEANING_FLIP"
  link_why_contrast="$LINK_WHY_CONTRAST" link_why_hard_negative=4 link_why_alone="$LINK_WHY_ALONE"
  pointers=0.1 entities=0.05 hindsight=0.1 near_misses=0.2 near_miss_margin=1   # run C
  residual=0.01 variance=0.1 absence=0.2 adversary=0.1 policy_l1=0   # 0 while there is one company: L1 only collapsed alpha
  explanation_only=0.5 changes=0.3 claims=0.3 rule_links=0.5 why_echo=0.2 negatives=0.3
  clause_use=0
  citations="$CITATIONS" derived_cards="$DERIVED_CARDS" table_kind="$TABLE_KIND"
  coherence="$COHERENCE" node_pointers="$NODE_POINTERS"
)
loss_args=()
for kv in "${weights[@]}"; do loss_args+=(--loss-weight "$kv"); done
ramp_args=()
if [[ "$RAMP" != 0 ]]; then
  for t in why_contrast why_alone why_swap keyword_dropout layout_lock entity_swap meaning_flip link_why_contrast link_why_alone \
           citations derived_cards table_kind coherence node_pointers; do ramp_args+=(--ramp "$t=$RAMP"); done
  ramp_args+=(--ramp numbers=1)   # the numbers stage opens on the new hours heads
fi

# why2 (ml/c3/why2.py, free_rows.py; arch_review §5-8). The critical heads' free
# rows read r only (FREE_R_ONLY; their h share ramps 1 -> 0 over RAMP passes via
# free_h_out), the described path, the card notes and the [purtera] lines train
# on their own, the company gradient into the trunk is scaled (COMPANY_R_GRAD).
# Every term ramps from 0 over RAMP passes; why_swap over 2 x RAMP, after
# why_contrast has placed r near the WHYs. Card heads' terms open with the cards.
DESCRIBED_ALONE=${DESCRIBED_ALONE:-0.3} CARD_ALONE=${CARD_ALONE:-0.2} CARD_CONTRAST=${CARD_CONTRAST:-0.2}
CO_WHY_CONTRAST=${CO_WHY_CONTRAST:-0.2} CO_WHY_ALONE=${CO_WHY_ALONE:-0.2}
NUMBER_WHY_ALONE=${NUMBER_WHY_ALONE:-0.1} JUDGMENT_WHY_ALONE=${JUDGMENT_WHY_ALONE:-0.2}
COMPANY_R_GRAD=${COMPANY_R_GRAD:-0.1} FREE_R_ONLY=${FREE_R_ONLY:-1}
why2=(
  described_alone="$DESCRIBED_ALONE" described_alone_p=0.5
  card_alone="$CARD_ALONE" card_contrast="$CARD_CONTRAST" card_cap=256
  co_why_contrast="$CO_WHY_CONTRAST" co_why_alone="$CO_WHY_ALONE"
  number_why_alone="$NUMBER_WHY_ALONE" judgment_why_alone="$JUDGMENT_WHY_ALONE"
  free_h_out=1 company_r_grad="$COMPANY_R_GRAD"
)
for kv in "${why2[@]}"; do loss_args+=(--loss-weight "$kv"); done
if [[ "$RAMP" != 0 ]]; then
  for t in described_alone card_alone card_contrast co_why_contrast co_why_alone number_why_alone \
           judgment_why_alone free_h_out; do ramp_args+=(--ramp "$t=$RAMP"); done
  ramp_args+=(--ramp "why_swap=$((2 * RAMP))")
fi
# Free rows: r only on the critical heads (free_rows.R_ONLY_KEYS); company rows
# on co_action / co_reason, so the company layer can be refit alone.
why2_free=()
if [[ "$FREE_R_ONLY" == 1 ]]; then
  for k in col:label_type col:admission col:supplier col:about read:site jdg:site_role jdg:site_role:reason \
           read:sow_section read:sow_element read:task_owner_role; do
    why2_free+=(--free-head "r:$k")
  done
fi
why2_free+=(--free-head read:co_action --free-head read:co_reason)

# Lines per part from the card's memory and the graphs each step holds.
graphs=1
[[ "$LAYOUT_LOCK" != 0 ]] && graphs=$((graphs + 1))
[[ "$KEYWORD_DROPOUT" != 0 ]] && graphs=$((graphs + 1))
[[ "$ENTITY_SWAP" != 0 ]] && graphs=$((graphs + 1))
[[ "$MEANING_FLIP" != 0 ]] && graphs=$((graphs + 1))
if [[ -z "${ATOMS:-}" ]]; then
  if [[ "$TINY" == 1 ]]; then ATOMS=512
  else
    mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' ')
    if (( ${mib:-0} >= 120000 )); then big=8192
    elif (( ${mib:-0} >= 70000 )); then big=4096
    else echo "GPU 0 has ${mib:-no} MiB: the 2-block encoder run needs an 80 GB card (set ATOMS to force)" >&2; exit 1
    fi
    ATOMS=$(( big / graphs / 512 * 512 ))
  fi
fi
echo "atoms per part $ATOMS ($graphs graphs per step)"
if [[ "$TINY" == 1 ]]; then VALID_ATOMS=${VALID_ATOMS:-$ATOMS}; else VALID_ATOMS=${VALID_ATOMS:-8192}; fi
echo "validation window $VALID_ATOMS (serve reads 8192)"

args=(--max-atoms "$ATOMS" --train "${train_files[@]}" --validation "${valid_files[@]}"
      --lr "${LR:-1e-4}" --warmup-passes "${WARMUP:-1}"
      "${balance_args[@]}"
      --free-head all "${head_weight_args[@]}"
      "${loss_args[@]}" ${ramp_args[@]+"${ramp_args[@]}"}
      --keep-checkpoints "$KEEP" --pick-floors "$FLOORS"
      --validation-max-atoms "$VALID_ATOMS" --deal-rate "${DEAL_RATE:-0.3}")
graph_args=()
[[ "$GRAPH" != off ]] && graph_args=(--graph "$GRAPH")
args+=(${graph_args[@]+"${graph_args[@]}"})
# The heads INIT does not have: derived at launch from the schema diff against
# INIT's schema (ml/c3/new_heads.py): INIT_SCHEMA_REF when set, else the saved
# run C shape (INIT_SHAPE, ml/c3/runE/init_shape.json, ref 0b6efce). Fails when
# a difference is not a head addition, or INIT was not saved under that schema.
# NEW_HEADS set by hand is checked against the derived list.
INIT_SHAPE=${INIT_SHAPE:-$REPO/ml/c3/runE/init_shape.json}
new_heads=""
if [[ "$TINY" != 1 ]]; then
  if [[ -n "${INIT_SCHEMA_REF:-}" ]]; then shape_src=(--ref "$INIT_SCHEMA_REF"); else shape_src=(--shape "$INIT_SHAPE"); fi
  derived=$(python -m ml.c3.new_heads "${shape_src[@]}" --checkpoint "$INIT") || {
    echo "NO-GO: the schema is not INIT's plus new heads (ml/c3/new_heads.py)" >&2; exit 1; }
  if [[ -n "${NEW_HEADS:-}" ]]; then
    a=$(tr ',' '\n' <<< "$NEW_HEADS" | sort | tr '\n' ' '); b=$(tr ',' '\n' <<< "$derived" | sort | tr '\n' ' ')
    [[ "$a" == "$b" ]] || { echo "NEW_HEADS ($NEW_HEADS) is not the derived list ($derived)" >&2; exit 1; }
  fi
  new_heads=$derived
  echo "new heads since INIT: $new_heads"
fi
anchor_heads=$(python - "$ANCHOR_HEADS" "$new_heads" <<'PY'
import sys
new = set(filter(None, sys.argv[2].split(",")))
print(",".join(k for k in sys.argv[1].split(",") if k and k not in new))
PY
)
args+=(--trunk-lr-scale "$TRUNK_LR_SCALE" --trunk-pull "$TRUNK_PULL" --anchor-kl "$ANCHOR_KL"
       --anchor-heads "$anchor_heads")
case "${VIEW_TAG:-1}" in
  1) args+=(--view-tag) ;;
  0) args+=(--no-view-tag) ;;
  *) echo "VIEW_TAG must be 0 or 1" >&2; exit 1 ;;
esac
echo "trunk lr x$TRUNK_LR_SCALE, pull $TRUNK_PULL/step to INIT, KL $ANCHOR_KL on $anchor_heads; graph $GRAPH"
# Pilot A M2: a text change is a breaking change until INIT reads as well under it.
if [[ "$TINY" != 1 && "${DESC_GATE:-1}" == 1 ]]; then
  # DESC_OLD_CK: when INIT is a head_fit refit on the new texts, the checkpoint it was refit from.
  gate_args=(--checkpoint "$INIT" ${DESC_OLD_CK:+--old-checkpoint "$DESC_OLD_CK"} --deal "${valid_files[@]}" ${DESC_GATE_DEALS:-}
             --max-drop "${DESC_MAX_DROP:-0.02}" --device "${C3_DEVICE:-cuda}" ${ENCODER:+--encoder "$ENCODER"} --json "$C3/out/desc_gate.json")
  [[ -n "${DESC_GATE_DRAFTS:-}" ]] && gate_args+=(--drafts-gold $DESC_GATE_DRAFTS)
  [[ "$GRAPH" != off ]] && gate_args+=(--graph-set "$GRAPH")
  set +e
  python -m ml.c3.desc_gate "${gate_args[@]}" 2>&1 | tee "$C3/out/desc_gate.log"
  s=${PIPESTATUS[0]}
  set -e
  (( s == 0 )) || { echo "NO-GO: the new texts alone cost INIT more than ${DESC_MAX_DROP:-0.02} on a head ($C3/out/desc_gate.json); refit those heads (ml/c3/head_fit.py) first" >&2; exit 1; }
fi
if [[ "$TINY" == 1 ]]; then
  args+=(--tiny)
else
  args+=(--init-from "$INIT" --allow-text-change --allow-new-heads "$new_heads"
         --init-schema-ref "${INIT_SCHEMA_REF:-$INIT_SHAPE}"
         --train-encoder-layers "${ENC_LAYERS:-2}" --encoder-lr "${ENCODER_LR:-2e-5}"
         ${ENCODER:+--encoder "$ENCODER"})
fi
stages="lines=${LINES:-12},links=${LINKS:-6},cards=${CARDS:-6},numbers=${NUMBERS:-2},deal=${DEAL:-2},open_links=${OPEN_LINKS:-4}"
TIER_GATE=${TIER_GATE:-0.98}
gates=()
args+=("${why2_free[@]}")   # after --free-head all: the r: rows convert the plain ones
[[ "${LINES_GATE:-0.50}" != 0 ]] && gates+=(--gate "lines@2.valid_exact>=${LINES_GATE:-0.50}")
# A start that does not reproduce run C's check means the files or the part
# size moved; only comparable at run C's 8192-line validation window and run
# C's mask (the view mask lets the start's attention read more than it trained on).
[[ "$VALID_ATOMS" == 8192 && ( "$GRAPH" == off || "$GRAPH" == *mask=index* ) ]] &&
  gates+=(--gate "start.valid_exact>=${START_FLOOR:-0.648}")
if [[ "$TEACH" == 1 ]]; then
  [[ -f "$TEACHER" ]] || { echo "no TEACHER $TEACHER (TEACH=0 skips the teach stage)" >&2; exit 1; }
  args+=(--teacher-from "$TEACHER" --teacher-steps 1
         --view-parts "${VIEW_PARTS:-24}" --lesson-gold "${LESSON_GOLD:-2}")
  [[ "${TEACHER_CONTINUE:-0}" == 1 ]] && args+=(--teacher-continue)
  stages+=",bind=1,teach=${TEACH_STEPS:-2}"
  gates+=(--gate "teach.valid_jdg_task_tier>=$TIER_GATE")
  for ((p = ${CHECK_EVERY:-2}; p < ${TEACH_STEPS:-2}; p += ${CHECK_EVERY:-2})); do
    gates+=(--gate "teach@$p.valid_jdg_task_tier>=$TIER_GATE")
  done
  if [[ "${FLIPS:-0}" == 1 ]]; then
    args+=(--extra-lessons); stages+=",flips=1,spread=0"
    gates+=(--gate "flips.valid_jdg_task_tier>=$TIER_GATE")
  else
    stages+=",flips=0,spread=0"
  fi
else
  args+=(--teacher-steps 0); stages+=",bind=0,teach=0,flips=0,spread=0"
fi
[[ -n "${ACCEPT_DRAFTS:-}" ]] && args+=(--accept-drafts "$ACCEPT_DRAFTS")
args+=(--stage-steps "$stages")
echo "stages $stages"
echo "commit $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"

if [[ "${SMOKE:-0}" == 1 ]]; then
  # The pre-launch gates (on by default; SIGNAL_AUDIT=0 skips them, for a
  # memory-only smoke): the signal audit before the loop, the bypass and
  # portability checks on the smoke's checkpoint after it. The go/no-go
  # numbers: $C3/out/smoke_gate.md.
  model_args=()
  if [[ "$TINY" == 1 ]]; then model_args=(--tiny)
  else
    model_args=(--allow-new-heads "$new_heads" --train-encoder-layers 2
                --init-schema-ref "${INIT_SCHEMA_REF:-$INIT_SHAPE}")
  fi
  if [[ "${SIGNAL_AUDIT:-1}" == 1 ]]; then
    # Every labeled signal and every kind of reasoning text must reach the
    # step's gradient (ml/c3/signal_audit.py).
    audit=(--train "${train_files[@]}" --max-atoms "$ATOMS" "${loss_args[@]}"
           "${head_weight_args[@]}" --stage-steps "$stages"
           --min-count "${SIGNAL_MIN:-50}" --allowlist "${SIGNAL_ALLOWLIST:-$REPO/ml/c3/runE/signal_allowlist.json}"
           --min-kind-words "${SIGNAL_KIND_WORDS:-10000}" --min-coverage "${SIGNAL_COVERAGE:-0.95}"
           --out "$C3/out/signal_audit.md" --json "$C3/out/signal_audit.json" "${model_args[@]}"
           "${why2_free[@]}" ${graph_args[@]+"${graph_args[@]}"})
    [[ -n "${ACCEPT_DRAFTS:-}" ]] && audit+=(--accept-drafts "$ACCEPT_DRAFTS")
    [[ "$TINY" != 1 ]] && audit+=(--init-from "$INIT")
    set +e
    python -m ml.c3.signal_audit "${audit[@]}" 2>&1 | tee "$C3/out/signal_audit.log"
    s=${PIPESTATUS[0]}
    set -e
    (( s == 0 )) || { echo "NO-GO: a labeled signal or reasoning text trains nothing ($C3/out/signal_audit.md)" >&2; exit 1; }
  fi
  export C3_PROGRESS_EVERY=1
  MEM=""
  if [[ "$TINY" != 1 ]] && command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 5 > "$C3/out/smokeE_mem.log" &
    MEM=$!
  fi
  set +e
  python -m ml.c3.loop "${args[@]}" --smoke --steps 1 --check-every 1 --checkpoint "$C3/out/smokeE.pt" \
    2>&1 | tee "$C3/out/smokeE.log"
  status=${PIPESTATUS[0]}
  if [[ -n "$MEM" ]]; then
    kill $MEM
    python - "$C3/out/smokeE_mem.log" <<'PY'
import sys, collections
peak = collections.defaultdict(int)
for line in open(sys.argv[1]):
    i, mib = [x.strip() for x in line.split(",")]
    peak[i] = max(peak[i], int(mib.split()[0]))
print("peak GPU memory MiB (nvidia-smi):", dict(peak))
PY
  fi
  if (( status == 0 )) && [[ "${SIGNAL_AUDIT:-1}" == 1 ]]; then
    # Critical heads decide from r, not the raw h path; a company refit step
    # leaves the base outputs bit-identical (ml/c3/smoke_checks.py).
    python -m ml.c3.smoke_checks --checkpoint "$C3/out/smokeE.pt" --train "${train_files[@]}" \
      --max-atoms "$ATOMS" --max-h-share "${MAX_H_SHARE:-0.10}" "${model_args[@]}" "${why2_free[@]}" \
      --free-h-out 1 ${graph_args[@]+"${graph_args[@]}"} \
      --signal-json "$C3/out/signal_audit.json" --out "$C3/out/smoke_gate.md" --json "$C3/out/smoke_gate.json" \
      2>&1 | tee "$C3/out/smoke_gate.log"
    status=${PIPESTATUS[0]}
    (( status == 0 )) || echo "NO-GO: smoke checks failed ($C3/out/smoke_gate.md)" >&2
  fi
  (( status == 0 )) && pick "$C3/out/smokeE.pt"
  exit "$status"
fi

# The run, then the pick and a slim copy of it for the Mac, all detached.
# RESUME=1 carries on from $CK.resume.pt (the loop writes it at every check).
[[ "${RESUME:-0}" == 1 ]] && args+=(--resume)
echo "restart after a failure: RESUME=1 with the same environment ($(env | grep -E '^(TRAIN_DIR|VALID_FILE|ATOMS|LINES|LINKS|CARDS|NUMBERS|OPEN_LINKS|TEACH|LAYOUT_LOCK|KEYWORD_DROPOUT|ENTITY_SWAP|MEANING_FLIP|WHY_[A-Z]+|LINK_WHY_[A-Z]+|TYPE_WEIGHT|RAMP|KEEP|CK|INIT|ACCEPT_DRAFTS)=' | tr '\n' ' '))bash $0"
run() {
  set +e
  python -m ml.c3.loop "${args[@]}" --steps 1 --check-every "${CHECK_EVERY:-2}" ${gates[@]+"${gates[@]}"} \
    --checkpoint "$CK" >> "$C3/out/runE.log" 2>&1
  echo "loop exit $?" >> "$C3/out/runE.log"
  pick "$CK" >> "$C3/out/runE.log" 2>&1 &&
    best=$(readlink "$CK.best") &&
    if [[ "$best" == "$(readlink -f "$INIT")" ]]; then echo "pick is the start: nothing to slim" >> "$C3/out/runE.log"
    elif [[ "$TINY" == 1 ]]; then echo "tiny: no slim" >> "$C3/out/runE.log"
    else python -m ml.c3.slim --checkpoint "$best" --out "$CK.best.slim" >> "$C3/out/runE.log" 2>&1
      refit "$CK.best.slim"; fi
}
# The Questions overlay reads the base's pooled features: one fitted on run C
# is stale on any run E checkpoint. Refit it on the pick (its md5 goes in the
# file; serve and gate.sh refuse it on another base), or REFIT=0 and serve
# without SOWSMITH_C3_QUESTION_REFIT.
refit() {
  [[ "${REFIT:-1}" == 1 ]] || { echo "REFIT=0: serve this checkpoint without the question overlay" >> "$C3/out/runE.log"; return; }
  SOWSMITH_C3_CHECKPOINT="$1" SOWSMITH_C3_ENCODER="${ENCODER:-answerdotai/ModernBERT-large}" \
    python -m ml.c3.question_fit --deals "${train_files[@]}" "${valid_files[@]}" --out "$1.question_refit.pt" \
    >> "$C3/out/runE.log" 2>&1 || echo "question refit failed: serve without the overlay" >> "$C3/out/runE.log"
}
[[ "${RESUME:-0}" == 1 ]] || : > "$C3/out/runE.log"
if [[ "${FG:-0}" == 1 ]]; then run; exit; fi
( trap '' HUP; run ) < /dev/null > /dev/null 2>&1 &
echo "Run E pid $!  log $C3/out/runE.log  pick $CK.pick.json  slim $CK.best.slim"
