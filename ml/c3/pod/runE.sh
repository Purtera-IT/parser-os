#!/usr/bin/env bash
# Run E: the parts run C switched off, turned back on, from the live checkpoint.
# Run C froze the encoder, skipped the teacher and trained lines and open_links
# only. Here the top 2 encoder blocks learn (run B's setting, at 2e-5), links
# and cards train, and the teach stage runs last, gated so it cannot cost task
# tier (run A's teach and run D's teach@1 both broke jdg:task_tier: 1.000 to
# .656 / .545). After the loop a pick step keeps the best checkpoint whose
# check held valid_exact and valid_jdg_task_tier against the run's own start
# check (= the live model): a failed teach stage costs nothing but its time.
#
# The layout meaning lock is on: the run E files carry layout_rewrites on table
# rows, and losses.layout_lock_loss (runE-data, 2eaa5ef) reads each part once
# more with one rewrite per row and holds the saved answers and the real-layout
# reading. No teacher needed. It is one more forward and backward on every part
# with rewrites (most of them), so about twice the activation memory and ~1.8x
# the student time; ATOMS defaults halve for it. The teacher's own rewrites
# (supercharge.teach_rewrites) stay unwired in loop.py. FLIPS=1 adds the flips
# lesson (teach_flip at 0.2).
#
#   TRAIN_DIR   training files (default $C3/runE/train); several dirs, space
#               separated, train together (a quoting run: "$C3/runE/train_presow
#               $C3/runE/train_estimator"). Every NNNNNN.json or NNNNNN@<view>.json
#               there trains, minus the validation deal and the fixed holdouts,
#               unless TRAIN_DEALS lists the file names (without .json). A listed
#               validation or holdout deal is refused, and every chosen file's
#               own deal_id is checked again before the run
#   VALID_FILE  the validation deal (default <first TRAIN_DIR>/010043.json, else
#               $C3/train/010043.json); never trained, in any view
#   INIT        start checkpoint (default $C3/runC.pt.open_links.slim, the live
#               one; its md5 is checked unless INIT_MD5=skip)
#   LAYOUT_LOCK weight of layout_lock (0.5); 0 turns it off
#   ATOMS       lines per part. Default from the card. Run B's 2-block student
#               + teacher peaked at 113-124 GB at 8192 on an H200, and the lock
#               doubles activations: with the lock 4096 at >=120 GB (H200), 2048
#               at >=70 GB (H100 / A100 80 GB); without it 8192 / 4096. Under
#               70 GB it refuses. Prove the peak with SMOKE=1 C3_SMOKE_LARGEST=1
#   LINES LINKS CARDS NUMBERS OPEN_LINKS   passes per student stage (12 6 6 2 4);
#               numbers trains the hours value and the hours per / qty heads
#   ACCEPT_DRAFTS the owner's approval manifest ({"files": [{"file", "sha256"}]}):
#               a file of assistant drafts only (runE/train_estimator) trains
#               only when listed here; otherwise it is left out
#   INIT_SCHEMA_REF  the commit INIT was trained on, when INIT predates shape
#               digests (loop.py --init-schema-ref). The init allows text changes
#               and the two added heads jdg:hours:per / jdg:hours:qty
#   TEACH=0     no bind/teach and the 7B is never loaded (student stages only)
#   TEACHER     saved teacher (default $C3/runA.pt.teacher)
#   TEACHER_CONTINUE=1  one more teacher pass first (+~2.6 h on an H200)
#   TEACH_STEPS VIEW_PARTS LESSON_GOLD   teach passes (2), viewed parts (24),
#               gold steps per lesson step on unviewed parts (2), as run D
#   TIER_GATE   the in-loop stop for teach/flips: valid_jdg_task_tier floor (0.98)
#   EXACT_TOL TIER_TOL  the pick's allowance below the start check (0, 0.01)
#   CHECK_EVERY checks (and a saved copy, ~3.4 GB) every N passes (2)
#   SMOKE=1     every stage the run trains, once on two parts, no gates, peak
#               GPU memory printed; with C3_SMOKE_LARGEST=1 the largest part
#   PICK_ONLY=1 only the pick step, on an existing $C3/out/runE.pt.checks.json
source "$(dirname "$0")/common.sh"
cd "$REPO"
CK=${CK:-$C3/out/runE.pt}
INIT=${INIT:-$C3/runC.pt.open_links.slim}
LIVE_MD5=5f49c472d5cbd1ff937cad2e7764c0a5   # c3_base/MANIFEST.json: runC.pt.open_links.slim
TEACHER=${TEACHER:-$C3/runA.pt.teacher}
TEACH=${TEACH:-1}
EXACT_TOL=${EXACT_TOL:-0} TIER_TOL=${TIER_TOL:-0.01}

# The best checkpoint whose check holds valid_exact and valid_jdg_task_tier
# against the start check; ties go to the later file. Writes <ck>.pick.json and
# links <ck>.best; the start (the live model) is the fallback.
pick() {
  python - "$1" "$INIT" "$EXACT_TOL" "$TIER_TOL" <<'PY'
import json, os, sys
ck, init, tol_e, tol_t = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
checks = json.load(open(f"{ck}.checks.json"))
if "start" not in checks:
    raise SystemExit("pick: no start check in " + ck + ".checks.json")
s = checks["start"]
floor_e = s["valid_exact"] - tol_e
floor_t = s.get("valid_jdg_task_tier", 0.0) - tol_t
rows = []
for at, row in checks.items():
    path = init if at == "start" else f"{ck}.{at}"
    ex, tier = row.get("valid_exact"), row.get("valid_jdg_task_tier")
    ok = ex is not None and tier is not None and ex >= floor_e and tier >= floor_t
    here = os.path.exists(path)
    rows.append((ok and here, ex or 0.0, os.path.getmtime(path) if here else 0.0, at, path, tier))
    print(f"pick {at:16} valid_exact={ex if ex is None else round(ex, 4)} "
          f"task_tier={tier if tier is None else round(tier, 4)} "
          f"{'pass' if ok else 'fail'}{'' if here else ' (no file)'}")
best = max(r for r in rows if r[0] or r[3] == "start")
print(f"pick floors valid_exact>={floor_e:.4f} task_tier>={floor_t:.4f}")
print(f"picked {best[3]} {best[4]} valid_exact={best[1]:.4f} task_tier={best[5]}")
json.dump({"at": best[3], "path": best[4], "valid_exact": best[1], "valid_jdg_task_tier": best[5],
           "floor_exact": floor_e, "floor_tier": floor_t}, open(f"{ck}.pick.json", "w"), indent=1)
link = f"{ck}.best"
if os.path.lexists(link):
    os.remove(link)
os.symlink(os.path.realpath(best[4]), link)
PY
}
if [[ "${PICK_ONLY:-0}" == 1 ]]; then pick "$CK"; exit; fi

# Training and validation files. A file is NNNNNN.json or a view of the deal,
# NNNNNN@<view>.json; the deal number before the @ decides what may train.
TRAIN_DIR=${TRAIN_DIR:-$C3/runE/train}
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
VALID_FILE=${VALID_FILE:-${dirs[0]}/$VALIDATION.json}
[[ -f "$VALID_FILE" ]] || VALID_FILE=$C3/train/$VALIDATION.json
echo "train ${#train_files[@]} files from $TRAIN_DIR: $(for f in "${train_files[@]}"; do basename "$f" .json; done | tr '\n' ' ')"
echo "validation $VALID_FILE"

# The start checkpoint is the live one.
[[ -f "$INIT" ]] || { echo "no INIT $INIT" >&2; exit 1; }
if [[ "${INIT_MD5:-}" != skip ]]; then
  if command -v md5sum >/dev/null; then got=$(md5sum "$INIT" | cut -d' ' -f1); else got=$(md5 -q "$INIT"); fi
  [[ "$got" == "$LIVE_MD5" ]] || { echo "INIT md5 $got is not the live checkpoint ($LIVE_MD5); INIT_MD5=skip to start elsewhere" >&2; exit 1; }
fi

# Lines per part from the card's memory.
if [[ -z "${ATOMS:-}" ]]; then
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -1 | tr -d ' ')
  big=8192; [[ "${LAYOUT_LOCK:-0.5}" != 0 ]] && big=4096
  if (( ${mib:-0} >= 120000 )); then ATOMS=$big
  elif (( ${mib:-0} >= 70000 )); then ATOMS=$((big / 2))
  else echo "GPU 0 has ${mib:-no} MiB: the 2-block encoder run needs an 80 GB card (set ATOMS to force)" >&2; exit 1
  fi
fi
echo "atoms per part $ATOMS"

args=(--max-atoms "$ATOMS" --train "${train_files[@]}" --validation "$VALID_FILE"
      --init-from "$INIT" --allow-text-change --allow-new-heads jdg:hours:per,jdg:hours:qty
      ${INIT_SCHEMA_REF:+--init-schema-ref "$INIT_SCHEMA_REF"}
      --train-encoder-layers 2 --encoder-lr "${ENCODER_LR:-2e-5}"
      --lr "${LR:-1e-4}" --warmup-passes "${WARMUP:-1}"
      --free-head all --head-weight col:label_type=20
      --loss-weight why_align=0.1 --loss-weight entities=0.05 --loss-weight hindsight=0.1
      --loss-weight pointers=0.1 --loss-weight near_misses=0.2
      --loss-weight layout_lock="${LAYOUT_LOCK:-0.5}")
stages="lines=${LINES:-12},links=${LINKS:-6},cards=${CARDS:-6},numbers=${NUMBERS:-2},deal=0,open_links=${OPEN_LINKS:-4}"
TIER_GATE=${TIER_GATE:-0.98}
gates=(--gate 'lines@2.valid_exact>=0.50')
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

if [[ "${SMOKE:-0}" == 1 ]]; then
  export C3_PROGRESS_EVERY=1
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 5 > "$C3/out/smokeE_mem.log" &
  MEM=$!
  set +e
  python -m ml.c3.loop "${args[@]}" --smoke --steps 1 --checkpoint "$C3/out/smokeE.pt" \
    2>&1 | tee "$C3/out/smokeE.log"
  status=${PIPESTATUS[0]}
  kill $MEM
  python - "$C3/out/smokeE_mem.log" <<'PY'
import sys, collections
peak = collections.defaultdict(int)
for line in open(sys.argv[1]):
    i, mib = [x.strip() for x in line.split(",")]
    peak[i] = max(peak[i], int(mib.split()[0]))
print("peak GPU memory MiB:", dict(peak))
PY
  (( status == 0 )) && pick "$C3/out/smokeE.pt"
  exit "$status"
fi

# The run, then the pick and a slim copy of it for the Mac, all detached.
(
  trap '' HUP
  set +e
  python -m ml.c3.loop "${args[@]}" --steps 1 --check-every "${CHECK_EVERY:-2}" "${gates[@]}" \
    --checkpoint "$CK" > "$C3/out/runE.log" 2>&1
  echo "loop exit $?" >> "$C3/out/runE.log"
  pick "$CK" >> "$C3/out/runE.log" 2>&1 &&
    best=$(readlink "$CK.best") &&
    if [[ "$best" == "$(readlink -f "$INIT")" ]]; then echo "pick is the start: nothing to slim" >> "$C3/out/runE.log"
    else python -m ml.c3.slim --checkpoint "$best" --out "$CK.best.slim" >> "$C3/out/runE.log" 2>&1; fi
) < /dev/null > /dev/null 2>&1 &
echo "Run E pid $!  log $C3/out/runE.log  pick $CK.pick.json  slim $CK.best.slim"
