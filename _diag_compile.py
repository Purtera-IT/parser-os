"""Run compile_brief.py inside the container with stderr-to-stdout.

This is a one-shot diagnostic to find why PM_HANDOFF.json isn't being
written. We download the latest envelope.json for OPTBOT, then run
compile_brief.py with the same args the worker uses but with stderr
captured to stdout so we can see the traceback that's currently being
discarded.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from azure.storage.blob import BlobServiceClient

DEAL_ID = "841ea7e0-0e2f-412a-aebc-5794c199b85c"
CONN = os.environ["ORBITBRIEF_ARTIFACTS_CONNECTION_STRING"]
CONTAINER = os.environ.get("ORBITBRIEF_ARTIFACTS_CONTAINER", "orbitbrief-artifacts")
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "https://ollama-mac-proxy-dev-eus2.whitehill-a3348ba5.eastus2.azurecontainerapps.io",
)
CHAT_MODEL = os.environ.get("ORBITBRIEF_CHAT_MODEL", "qwen3:14b")
CORE_ROOT = Path(os.environ.get("ORBITBRIEF_CORE_ROOT", "/app/Orbitbrief-Core"))
PARSER_ROOT = Path(os.environ.get("PARSER_OS_ROOT", "/app/parser-os"))

client = BlobServiceClient.from_connection_string(CONN)

work = Path(tempfile.mkdtemp(prefix="ob-diag-"))
envelope_path = work / "envelope.json"
out_dir = work / "out"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"[diag] download envelope -> {envelope_path}", flush=True)
blob = f"deals/{DEAL_ID}/orbitbrief/latest/envelope.json"
with open(envelope_path, "wb") as f:
    client.get_blob_client(CONTAINER, blob).download_blob().readinto(f)
print(f"[diag] envelope downloaded {envelope_path.stat().st_size} bytes", flush=True)

py_path = os.pathsep.join(
    [str(PARSER_ROOT), str(CORE_ROOT / "src"), os.environ.get("PYTHONPATH", "")]
).strip(os.pathsep)

cmd = [
    sys.executable,
    str(CORE_ROOT / "compile_brief.py"),
    str(envelope_path),
    "--out",
    str(out_dir),
    "--quiet-parser",
    "--ollama",
    "--ollama-base-url",
    OLLAMA_BASE_URL,
    "--chat-model",
    CHAT_MODEL,
]
env = {**os.environ, "PYTHONPATH": py_path}
print(f"[diag] cmd: {cmd}", flush=True)
print(f"[diag] cwd: {CORE_ROOT}", flush=True)

# stderr=STDOUT so we see EVERYTHING
proc = subprocess.run(
    cmd,
    cwd=str(CORE_ROOT),
    env=env,
    capture_output=True,
    text=True,
    timeout=int(os.environ.get("COMPILE_TIMEOUT_SEC", "840")),
)

print(f"[diag] returncode: {proc.returncode}", flush=True)
print(f"[diag] === STDOUT ===", flush=True)
print(proc.stdout, flush=True)
print(f"[diag] === STDERR ===", flush=True)
print(proc.stderr, flush=True)

print(f"[diag] === OUT DIR CONTENTS ===", flush=True)
for p in sorted(out_dir.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size} bytes)", flush=True)

handoff = out_dir / "PM_HANDOFF.json"
print(f"[diag] PM_HANDOFF.json exists: {handoff.is_file()}", flush=True)

# cleanup
shutil.rmtree(work, ignore_errors=True)
