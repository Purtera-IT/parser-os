#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="/tmp/purtera_mvp_demo"
mkdir -p "${DEMO_DIR}"

echo "[1/7] Regenerating fixtures..."
python scripts/make_demo_fixtures.py

echo "[2/7] Running targeted regression tests..."
pytest tests/test_schemas.py tests/test_packetizer.py tests/test_quality_gates.py

echo "[3/7] Compiling demo project..."
python -m app.cli compile tests/fixtures/demo_project \
  --allow-unverified-receipts \
  --out "${DEMO_DIR}/compile_result.json" \
  --trace-out "${DEMO_DIR}/trace.json"

echo "[4/7] Running packetizer benchmark..."
if ! python scripts/run_packetizer_benchmark.py \
  --fixtures tests/fixtures/gold_scenarios \
  --out "${DEMO_DIR}/packetizer_benchmark.json"; then
  echo "Packetizer benchmark thresholds failed; continuing demo generation with captured report."
fi

echo "[5/7] Running adversarial lab..."
python scripts/run_adversarial_lab.py \
  --count 25 \
  --out "${DEMO_DIR}/adversarial_report.json"

echo "[6/7] Running parser benchmark..."
if ! python scripts/run_parser_benchmark.py \
  --out "${DEMO_DIR}/parser_benchmark.json"; then
  echo "Parser benchmark thresholds failed; continuing demo generation with captured report."
fi

echo "[7/7] Building demo markdown reports..."
python scripts/build_demo_report.py --demo-dir "${DEMO_DIR}"

echo "Demo outputs ready in ${DEMO_DIR}"
