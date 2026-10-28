#!/usr/bin/env bash
# Pack EXACTLY what the Mac C3 server runs into c3_base.tgz for the GPU image, so the T4 answers
# like the Mac: the live checkpoint, encoder, question refit and quote-heads bundle (~/c3-serve/c3_base)
# and the live code working copy (~/c3-serve/code: ml/ and c3_record/ .py + .json only, no caches, no deal dumps).
# Writes the file only; uploading it is a separate, deliberate step.
#   OUT=~/c3_base_t4.tgz bash ml/c3/serve_image/pack_live.sh
set -euo pipefail
SRC=${C3_SERVE:-$HOME/c3-serve}
OUT=${OUT:-$HOME/c3_base_t4.tgz}
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/c3_base/code"
for f in checkpoint.pt MANIFEST.json question_refit.pt quote_heads_bundle.json; do
  [[ -f "$SRC/c3_base/$f" ]] && cp "$SRC/c3_base/$f" "$WORK/c3_base/"
done
cp -R "$SRC/c3_base/encoder" "$WORK/c3_base/encoder"
(cd "$SRC/code" && { find ml c3_record -name '*.py' -o -name '*.json'; find app/core -maxdepth 1 -name '*.json'; } | grep -v __pycache__ | tar -cf - -T -) | tar -xf - -C "$WORK/c3_base/code"
python3 - "$WORK/c3_base" <<'PY'
import hashlib,json,os,sys,subprocess
d=sys.argv[1]; src=os.path.expanduser(os.environ.get('C3_SERVE','~/c3-serve'))
m=json.load(open(os.path.join(d,'MANIFEST.json')))
m['packed_from']='live mac working copy'
m['code_head']=subprocess.run(['git','-C',os.path.join(src,'code'),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
m['code_dirty']=bool(subprocess.run(['git','-C',os.path.join(src,'code'),'status','--porcelain','ml'],capture_output=True,text=True).stdout.strip())
m['checkpoint_md5_packed']=hashlib.md5(open(os.path.join(d,'checkpoint.pt'),'rb').read()).hexdigest()
json.dump(m,open(os.path.join(d,'MANIFEST.json'),'w'),indent=1)
PY
tar -czf "$OUT" -C "$WORK" c3_base
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
