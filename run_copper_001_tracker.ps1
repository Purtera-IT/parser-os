$ErrorActionPreference = "Continue"

$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = "c:\Users\lilli\Downloads\purtera_copper_low_voltage_public_validation_packs\purtera_copper_low_voltage_validation_packs\real_data_cases"
$Case = "COPPER_001_SPRING_LAKE_AUDITORIUM"
$CaseDir = Join-Path $Root $Case
$OutDir = Join-Path $CaseDir "outputs"
$CompileResult = Join-Path $OutDir "compile_result.json"
$Benchmark = Join-Path $OutDir "benchmark_summary.json"
$Coverage = Join-Path $OutDir "coverage.json"
$ReviewQueue = Join-Path $OutDir "review_queue.json"

Set-Location $Repo

Write-Host "=== COPPER 001 VALIDATION TRACKER ==="
Write-Host "Repo: $Repo"
Write-Host "Root: $Root"
Write-Host "Case: $Case"
Write-Host ""

Write-Host "Step 1: Checking paths..."
if (!(Test-Path $Root)) { Write-Host "MISSING ROOT: $Root"; exit 1 }
if (!(Test-Path $CaseDir)) { Write-Host "MISSING CASE: $CaseDir"; exit 1 }
if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

Write-Host "OK paths exist."
Write-Host ""

Write-Host "Step 2: Compiling case..."
python scripts/compile_real_data_case.py --case-id $Case --root $Root

Write-Host ""
Write-Host "Step 3: Checking compile outputs..."
if (Test-Path $CompileResult) {
  Write-Host "FOUND compile_result.json"
} else {
  Write-Host "MISSING compile_result.json"
  exit 1
}

if (Test-Path $Benchmark) {
  Write-Host "FOUND benchmark_summary.json"
} else {
  Write-Host "MISSING benchmark_summary.json"
}

Write-Host ""
Write-Host "Step 4: Building coverage report..."
python scripts/evidence_coverage_report.py --compile-result $CompileResult --out $Coverage

Write-Host ""
Write-Host "Step 5: Building review queue..."
python scripts/build_review_queue.py --compile-result $CompileResult --out $ReviewQueue

Write-Host ""
Write-Host "Step 6: Quick JSON summary..."
$env:COPPER_TRACKER_ROOT = $Root
$env:COPPER_TRACKER_CASE = $Case
python -c @"
import json, pathlib, os
root = pathlib.Path(os.environ['COPPER_TRACKER_ROOT'])
case = os.environ['COPPER_TRACKER_CASE']
out = root / case / 'outputs'
cr = out / 'compile_result.json'
bm = out / 'benchmark_summary.json'
data = json.loads(cr.read_text(encoding='utf-8'))
atoms = data.get('atoms', [])
packets = data.get('packets', [])
manifest = data.get('manifest') or {}
fingerprints = manifest.get('artifact_fingerprints') or []
edges = data.get('edges', [])
print('Artifacts (manifest fingerprints):', len(fingerprints))
print('Atoms:', len(atoms))
print('Edges:', len(edges))
print('Packets:', len(packets))
families = {}
for p in packets:
    fam = p.get('family') or p.get('packet_family') or 'UNKNOWN'
    if isinstance(fam, dict):
        fam = fam.get('value', str(fam))
    families[str(fam)] = families.get(str(fam), 0) + 1
print('Packet families:')
for k, v in sorted(families.items()):
    print(f'  {k}: {v}')
if bm.exists():
    b = json.loads(bm.read_text(encoding='utf-8'))
    print('Benchmark keys:', ', '.join(sorted(b.keys())))
    if 'invalid_governance_count' in b:
        print('invalid_governance_count:', b['invalid_governance_count'])
"@

Write-Host ""
Write-Host "=== DONE ==="
Write-Host "Now inspect:"
Write-Host $CompileResult
Write-Host $Coverage
Write-Host $ReviewQueue
