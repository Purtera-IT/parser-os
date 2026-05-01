#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/tmp"
OUT_JSON="${OUT_DIR}/purtera_demo_output.json"

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

echo "[demo] regenerating fixtures"
python scripts/make_demo_fixtures.py

echo "[demo] running tests"
# Keep this enabled for local environments where external plugins crash pytest startup.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
pytest

echo "[demo] compiling demo project"
python -m app.cli compile tests/fixtures/demo_project --out "${OUT_JSON}"

echo "[demo] packet summary"
python scripts/inspect_packets.py "${OUT_JSON}"

echo "[demo] output: ${OUT_JSON}"
