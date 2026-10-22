#!/bin/bash
# refit.sh <out-name> [holdout-deal ...]
out=$1; shift
hold=$(IFS='|'; echo "${*:-NONE}")
cd ~/c3-serve/code
TR=$(ls ~/c3-local/train/*.json | grep -v '\.err')
SF=$(ls ~/c3-local/sowfix/train/*.json | grep -v -E "$hold")
PYTHONPATH=~/c3-serve/code exec python3.11 -m ml.c3.head_fit --checkpoint ~/c3-serve/c3_base/checkpoint.pt \
  --encoder ~/c3-serve/c3_base/encoder --train $TR $SF \
  --validation ~/c3-local/010087.json ~/c3-local/010149.json \
  --out ~/c3-local/sowfix/$out.pt --extra-keys read:co_action --device mps
