#!/usr/bin/env bash
set -u

OUT_DIR="/tmp/purtera_final_mvp"
mkdir -p "${OUT_DIR}"

run_step() {
  local name="$1"
  shift
  echo "[gauntlet] ${name}"
  if "$@"; then
    return 0
  fi
  echo "[gauntlet] WARNING: step failed: ${name}"
  return 1
}

echo "[gauntlet] Output directory: ${OUT_DIR}"

# 1) Regenerate fixtures
run_step "regenerate_fixtures" python scripts/make_demo_fixtures.py

# 2) Run full pytest and capture status
if python -m pytest -q; then
  python -c "import json; from pathlib import Path; Path('${OUT_DIR}/pytest_summary.json').write_text(json.dumps({'passed': True, 'command': 'python -m pytest -q'}, indent=2), encoding='utf-8')"
else
  python -c "import json; from pathlib import Path; Path('${OUT_DIR}/pytest_summary.json').write_text(json.dumps({'passed': False, 'command': 'python -m pytest -q'}, indent=2), encoding='utf-8')"
fi

# 3) Compile demo with security_camera pack
run_step "compile_demo_security_pack" python -m app.cli compile tests/fixtures/demo_project --domain-pack app/domain/security_camera_pack.yaml --allow-unverified-receipts --out "${OUT_DIR}/compile_result.json" --trace-out "${OUT_DIR}/trace.json"

# 4) Source replay validation summary
python -c "import json; from pathlib import Path; p=Path('${OUT_DIR}/compile_result.json'); payload=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}; atoms=payload.get('atoms', []); failed=0; verified=0; total=0; [None for a in atoms for r in (a.get('receipts') or []) if not (total:=total+1)]; failed=sum(1 for a in atoms for r in (a.get('receipts') or []) if r.get('replay_status')=='failed'); verified=sum(1 for a in atoms for r in (a.get('receipts') or []) if r.get('replay_status')=='verified'); out={'receipt_total': total, 'receipt_verified_count': verified, 'receipt_failed_count': failed, 'receipt_verification_rate': (verified/total if total else 0.0)}; Path('${OUT_DIR}/source_replay_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')"

# 5) Evidence coverage report
run_step "coverage_report" python scripts/evidence_coverage_report.py --compile-result "${OUT_DIR}/compile_result.json" --out "${OUT_DIR}/coverage.json"

# 6) Parser benchmark
if ! python scripts/run_parser_benchmark.py --allow-fail --out "${OUT_DIR}/parser_benchmark.json"; then
  echo "[gauntlet] parser benchmark exited non-zero (captured output)."
fi

# 7) Packetizer benchmark
if ! python scripts/run_packetizer_benchmark.py --allow-fail --fixtures tests/fixtures/gold_scenarios --out "${OUT_DIR}/packetizer_benchmark.json"; then
  echo "[gauntlet] packetizer benchmark exited non-zero (captured output)."
fi

# 8) Adversarial lab count 50
run_step "adversarial_lab" python scripts/run_adversarial_lab.py --count 50 --out "${OUT_DIR}/adversarial_report.json"

# 9) Domain pack certification
if ! python scripts/certify_domain_pack.py --allow-fail --domain-pack app/domain/security_camera_pack.yaml --fixtures tests/fixtures/gold_scenarios --out "${OUT_DIR}/domain_cert_security_camera.json"; then
  echo "[gauntlet] domain certification exited non-zero (captured output)."
fi

# 10) Semantic linker sandbox experiment
run_step "experiment_semantic_linker" python scripts/run_extraction_experiment.py --project tests/fixtures/demo_project --extractor semantic_linker --out "${OUT_DIR}/experiment_semantic_linker.json"

# 11) Fake LLM candidate sandbox experiment
run_step "experiment_llm_candidate" python scripts/run_extraction_experiment.py --project tests/fixtures/demo_project --extractor llm_candidate_extractor --out "${OUT_DIR}/experiment_llm_candidate.json"

# 12) Active learning queue
run_step "active_learning_queue" python scripts/build_review_queue.py --compile-result "${OUT_DIR}/compile_result.json" --out "${OUT_DIR}/active_learning_queue.json"

# 13) Perf benchmark (100 sites)
run_step "perf_benchmark_100_sites" python scripts/run_perf_benchmark.py --sites 100 --devices 1 --out "${OUT_DIR}/perf_100_sites.json"

# Optional: API hardening and real-data harness smoke (already in full pytest, but surfaced here)
if python -m pytest -q tests/test_api_hardening.py; then
  python -c "import json; from pathlib import Path; Path('${OUT_DIR}/api_hardening_summary.json').write_text(json.dumps({'passed': True}, indent=2), encoding='utf-8')"
else
  python -c "import json; from pathlib import Path; Path('${OUT_DIR}/api_hardening_summary.json').write_text(json.dumps({'passed': False}, indent=2), encoding='utf-8')"
fi
if python -m pytest -q tests/test_real_data_harness.py; then
  python -c "import json; from pathlib import Path; Path('${OUT_DIR}/real_data_harness_summary.json').write_text(json.dumps({'passed': True, 'skeleton_checked': True}, indent=2), encoding='utf-8')"
else
  python -c "import json; from pathlib import Path; Path('${OUT_DIR}/real_data_harness_summary.json').write_text(json.dumps({'passed': False, 'skeleton_checked': True}, indent=2), encoding='utf-8')"
fi

# Calibrator smoke (optional)
python -c "import json; from pathlib import Path; p=Path('${OUT_DIR}/compile_result.json'); payload=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}; packets=payload.get('packets') or []; rows=[]; 
from itertools import islice
for i,pkt in enumerate(islice(packets,0,2)): rows.append({'packet_id': pkt.get('id'), 'correct_packet': bool(i%2==0)})
Path('${OUT_DIR}/calibrator_labels.json').write_text(json.dumps({'reviews': rows}, indent=2), encoding='utf-8')"
if python scripts/train_calibrator.py "${OUT_DIR}/calibrator_labels.json" "${OUT_DIR}/calibrator.joblib" "${OUT_DIR}/compile_result.json"; then
  python scripts/evaluate_calibrator.py "${OUT_DIR}/calibrator.joblib" "${OUT_DIR}/compile_result.json" --abstain-threshold 0.7 > "${OUT_DIR}/calibrator_smoke.txt" || true
fi

# 14) Build final report
run_step "build_final_report" python scripts/build_final_mvp_report.py --out-dir "${OUT_DIR}"

echo "[gauntlet] Done. Artifacts in ${OUT_DIR}"
