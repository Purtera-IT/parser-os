#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
cd "$REPO"
id() { case $1 in 002|061|123) echo "000$1";; *) echo "010$1";; esac; }
rm -f "$C3/train/split.json"
for f in "$C3"/raw/export*.json; do
  n=$(basename "$f" .json); n=${n#export}; d=$(id "$n")
  split=train
  case " $VALIDATION " in *" $d "*) split=validation;; esac
  case " $HOLDOUT " in *" $d "*) split=holdout;; esac
  answers=()
  [ -f "$C3/raw/dealanswers$n.json" ] && answers=(--deal-answers "$C3/raw/dealanswers$n.json")
  python -m ml.c3.export --page "$f" --deal-id "$d" --output "$C3/train/$d.json" "${answers[@]}" \
    --manifest "$C3/train/split.json" --split "$split"
done
python - "$C3/train/split.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print({k: len(v) for k, v in m.items()})
assert {k: len(v) for k, v in m.items()} == {"train": 13, "validation": 3, "holdout": 4}, "expected 13/3/4 deals"
PY
