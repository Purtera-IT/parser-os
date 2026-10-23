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
# the graph count (the lock reuses the step's encoded descriptions). The teacher's own rewrites
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
#               one; its md5 is checked unless INIT_MD5=skip)
#   LAYOUT_LOCK KEYWORD_DROPOUT WHY_CONTRAST WHY_ALONE WHY_SWAP TYPE_WEIGHT RAMP
#   WHY_ALIGN WHY_SUFFICIENCY LINK_WHY_CONTRAST LINK_WHY_ALONE  override the
#               loss table (0.5 0.3 0.3 0.2 0 10 2 0 0 0.3 0.2); 0 turns a term off. The ablation configs are in loss_plan.md
#   ATOMS       lines per part. Default from the card and the graphs a step
#               holds (1 + layout_lock on + keyword_dropout on): 8192 / graphs at
#               >=120 GB (H200), 4096 / graphs at >=70 GB, rounded down to 512.
#               With both on: 2560 / 1024. Under 70 GB it refuses. Prove the
#               peak with SMOKE=1 C3_SMOKE_LARGEST=1 and set ATOMS from it
#   FLOORS      the live model's per-head scores on 010043 ($C3/runE/live_floors.json)
#   KEEP        checkpoints kept on disk, plus the best (3)
#   RESUME=1    carry on from $CK.resume.pt after a pod failure (same command
#               otherwise; see the restart line the launch prints)
#   TINY=1      the CPU stand-in (loop --tiny): no INIT, no GPU, ATOMS 512
#   FG=1        run in the foreground (tests), not detached
#   LINES_GATE  the lines@2 valid_exact stop (0.50); 0 turns it off
#   LINES LINKS CARDS NUMBERS OPEN_LINKS   passes per student stage (12 6 6 2 4);
#               numbers trains the hours value and the hours per / qty heads
#   ACCEPT_DRAFTS the owner's approval manifest ({"files": [{"file", "sha256"}]}):
#               a file of assistant drafts only (runE/train_estimator) trains
#               only when listed here; otherwise it is left out
#   INIT_SCHEMA_REF  the commit INIT was trained on, when INIT predates shape
#               digests (loop.py --init-schema-ref). The init allows text changes
#               and the two added heads jdg:hours:per / jdg:hours:qty
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
#   PICK_ONLY=1 only the pick step, on an existing $C3/out/runE.pt.checks.json
source "$(dirname "$0")/common.sh"
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
  python -m ml.c3.pick --checkpoint "$1" --init "$INIT" --floors "$FLOORS" --z "$PICK_Z" --gap "$PICK_GAP"
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
  if [[ "${INIT_MD5:-}" != skip ]]; then
    if command -v md5sum >/dev/null; then got=$(md5sum "$INIT" | cut -d' ' -f1); else got=$(md5 -q "$INIT"); fi
    [[ "$got" == "$LIVE_MD5" ]] || { echo "INIT md5 $got is not the live checkpoint ($LIVE_MD5); INIT_MD5=skip to start elsewhere" >&2; exit 1; }
  fi
fi

# The loss table (runE/loss_plan.md). Every LossWeights term is named, so a
# default changing in losses.py cannot change this run. A stage still trains
# only the terms its gold_weights turns on (relations from links, judgments from
# cards, numbers/literals from numbers, deal_numbers from deal/open_links).
LAYOUT_LOCK=${LAYOUT_LOCK:-0.5} KEYWORD_DROPOUT=${KEYWORD_DROPOUT:-0.3}
WHY_CONTRAST=${WHY_CONTRAST:-0.3} WHY_ALONE=${WHY_ALONE:-0.2} WHY_SWAP=${WHY_SWAP:-0}
TYPE_WEIGHT=${TYPE_WEIGHT:-10} RAMP=${RAMP:-2}
WHY_ALIGN=${WHY_ALIGN:-0} WHY_SUFFICIENCY=${WHY_SUFFICIENCY:-0}   # E-base ablation: 0.1 / 0.5 (run C)
LINK_WHY_CONTRAST=${LINK_WHY_CONTRAST:-0.3} LINK_WHY_ALONE=${LINK_WHY_ALONE:-0.2}   # link notes (link_why.py)
weights=(
  heads=1 conduct=1 numbers=0.2 relations=0.5 governs=0.5
  judgments=1 judgment_rationale=0.2 literals=0.4 deal_numbers=0.2
  why_align="$WHY_ALIGN" why_sufficiency="$WHY_SUFFICIENCY"   # 0: replaced by why_contrast / why_alone
  why_contrast="$WHY_CONTRAST" why_hard_negative=4
  why_alone="$WHY_ALONE"
  why_swap="$WHY_SWAP" why_swap_cap=64
  keyword_dropout="$KEYWORD_DROPOUT" keyword_dropout_p=0.3 keyword_dropout_cap=256
  layout_lock="$LAYOUT_LOCK"
  link_why_contrast="$LINK_WHY_CONTRAST" link_why_hard_negative=4 link_why_alone="$LINK_WHY_ALONE"
  pointers=0.1 entities=0.05 hindsight=0.1 near_misses=0.2 near_miss_margin=1   # run C
  residual=0.01 variance=0.1 absence=0.2 adversary=0.1 policy_l1=0.01
  explanation_only=0.5 changes=0.3 claims=0.3 rule_links=0.5 why_echo=0.2 negatives=0.3
  clause_use=0
)
loss_args=()
for kv in "${weights[@]}"; do loss_args+=(--loss-weight "$kv"); done
ramp_args=()
if [[ "$RAMP" != 0 ]]; then
  for t in why_contrast why_alone why_swap keyword_dropout layout_lock link_why_contrast link_why_alone; do ramp_args+=(--ramp "$t=$RAMP"); done
  ramp_args+=(--ramp numbers=1)   # the numbers stage opens on the new hours heads
fi

# Lines per part from the card's memory and the graphs each step holds.
graphs=1
[[ "$LAYOUT_LOCK" != 0 ]] && graphs=$((graphs + 1))
[[ "$KEYWORD_DROPOUT" != 0 ]] && graphs=$((graphs + 1))
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

args=(--max-atoms "$ATOMS" --train "${train_files[@]}" --validation "${valid_files[@]}"
      --lr "${LR:-1e-4}" --warmup-passes "${WARMUP:-1}"
      --free-head all --head-weight col:label_type="$TYPE_WEIGHT"
      "${loss_args[@]}" ${ramp_args[@]+"${ramp_args[@]}"}
      --keep-checkpoints "$KEEP" --pick-floors "$FLOORS")
if [[ "$TINY" == 1 ]]; then
  args+=(--tiny)
else
  args+=(--init-from "$INIT" --allow-text-change --allow-new-heads jdg:hours:per,jdg:hours:qty
         ${INIT_SCHEMA_REF:+--init-schema-ref "$INIT_SCHEMA_REF"}
         --train-encoder-layers 2 --encoder-lr "${ENCODER_LR:-2e-5}")
fi
stages="lines=${LINES:-12},links=${LINKS:-6},cards=${CARDS:-6},numbers=${NUMBERS:-2},deal=0,open_links=${OPEN_LINKS:-4}"
TIER_GATE=${TIER_GATE:-0.98}
gates=()
[[ "${LINES_GATE:-0.50}" != 0 ]] && gates+=(--gate "lines@2.valid_exact>=${LINES_GATE:-0.50}")
# A start that does not reproduce run C's check means the files or the part
# size moved; only comparable at run C's 8192 lines per part.
[[ "$ATOMS" == 8192 ]] && gates+=(--gate "start.valid_exact>=${START_FLOOR:-0.648}")
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
  (( status == 0 )) && pick "$C3/out/smokeE.pt"
  exit "$status"
fi

# The run, then the pick and a slim copy of it for the Mac, all detached.
# RESUME=1 carries on from $CK.resume.pt (the loop writes it at every check).
[[ "${RESUME:-0}" == 1 ]] && args+=(--resume)
echo "restart after a failure: RESUME=1 with the same environment ($(env | grep -E '^(TRAIN_DIR|VALID_FILE|ATOMS|LINES|LINKS|CARDS|NUMBERS|OPEN_LINKS|TEACH|LAYOUT_LOCK|KEYWORD_DROPOUT|WHY_[A-Z]+|LINK_WHY_[A-Z]+|TYPE_WEIGHT|RAMP|KEEP|CK|INIT|ACCEPT_DRAFTS)=' | tr '\n' ' '))bash $0"
run() {
  set +e
  python -m ml.c3.loop "${args[@]}" --steps 1 --check-every "${CHECK_EVERY:-2}" ${gates[@]+"${gates[@]}"} \
    --checkpoint "$CK" >> "$C3/out/runE.log" 2>&1
  echo "loop exit $?" >> "$C3/out/runE.log"
  pick "$CK" >> "$C3/out/runE.log" 2>&1 &&
    best=$(readlink "$CK.best") &&
    if [[ "$best" == "$(readlink -f "$INIT")" ]]; then echo "pick is the start: nothing to slim" >> "$C3/out/runE.log"
    elif [[ "$TINY" == 1 ]]; then echo "tiny: no slim" >> "$C3/out/runE.log"
    else python -m ml.c3.slim --checkpoint "$best" --out "$CK.best.slim" >> "$C3/out/runE.log" 2>&1; fi
}
[[ "${RESUME:-0}" == 1 ]] || : > "$C3/out/runE.log"
if [[ "${FG:-0}" == 1 ]]; then run; exit; fi
( trap '' HUP; run ) < /dev/null > /dev/null 2>&1 &
echo "Run E pid $!  log $C3/out/runE.log  pick $CK.pick.json  slim $CK.best.slim"
