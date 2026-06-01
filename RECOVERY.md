# 🚨 If something looks broken tomorrow — DO THIS

## Step 1: Run the smoke test (10 seconds)

```bash
cd ~/parser-os
./smoke_test.sh
```

This checks every part of the stack. If it ends with **🟢 ALL GREEN** — you're fine. The PM can use the system. Tell them to refresh the page.

If something shows red, go to Step 2.

## Step 2: Auto-recover (60 seconds)

```bash
./smoke_test.sh --fix
```

This restarts the Function App, clears poison queues, deletes zombie Tailscale nodes, and verifies recovery. If it ends 🟢 — you're back.

## Step 3: One-by-one recovery (if Step 2 didn't fix it)

| Symptom | Run this |
|---|---|
| Page returns 404 / "API stub" error | `az functionapp restart -n purpulse-dev-api-eus2 -g purtera-dev-rg` (wait 60s) |
| Worker stuck "queued" forever | `az containerapp job start -n parser-os-worker-dev-eus2 -g purtera-dev-rg` |
| Tailscale quota exhausted | `cd ~/parser-os && python _tailscale_bulk_delete.py --prefix parser-os-worker` |
| Function App totally broken | `az functionapp deploy -n purpulse-dev-api-eus2 -g purtera-dev-rg --src-path /tmp/wwwroot_v5710_real.zip --type zip --clean true --restart true` |
| Mac Studio (Ollama) offline | Tell Griffin to plug it in / check power; LLM stages will recover once it's reachable |

## Step 4: Nothing works — call for help

If smoke_test.sh keeps failing after `--fix`, take a screenshot of the output and send it. The stack is logged everywhere — Azure Monitor will show what failed.

---

## 📍 Known-good test URLs to give the PM

| Deal | Files | Expected output |
|---|---|---|
| https://id-preview--4ff57018-9974-43e7-ac39-e00dc74a8d9f.lovable.app/pm/quoting/1bf0c10e-e840-4a1f-b526-d8f417181ada?step=artifacts | 1 PDF (Notes) | 9 atoms · 1 entity |
| https://id-preview--4ff57018-9974-43e7-ac39-e00dc74a8d9f.lovable.app/pm/quoting/02557291-6d99-4adc-8d38-8c0802ecd35e?step=artifacts | 2 (DOCX + XLSX, real RFP) | 307 atoms · 72 entities |
| https://id-preview--4ff57018-9974-43e7-ac39-e00dc74a8d9f.lovable.app/pm/quoting/0504ccbe-c2fc-466b-a6d8-28eafcd49e08?step=artifacts | 3 (DOCX + XLSX + signed PDF) | 398 atoms · 46 entities — *archived in HubSpot, UI shows guard* |

---

## 💡 Things that ARE NOT broken even if they look weird

1. **EDGES: 0** with sub-label "single-file scope" — accurate, not a bug
2. **ENTITIES: 0** with sub-label "thin input" — also accurate
3. **Mac Studio shows OFFLINE in Tailscale status** but compile still works — Tailscale's online flag is sometimes stale; the compile itself proves connectivity
4. **First request takes 30 sec** — cold start, only happens after idle
5. **"This deal is not available in PurPulse"** — that deal is archived/closed in HubSpot; correct behavior

---

## 🛠 What I changed today (v57.x) for the record

- **v57.4** parser-os: split discovery-note Q&A blobs, suppress negated device entities
- **v57.5** Tailscale: ephemeral auth key + zombie cleanup
- **v57.6** parser-os-worker: URL-decode filenames with spaces / parens
- **v57.6** Function App: /resow writes real manifest blob
- **v57.9** UI: pipeline reads compile-progress, ingestion self-heals, badge un-clipped, ?step= deep links work
- **v57.10** Worker: timeout 30→60min, retries 0→1
- **v57.10** Function App: /rebrief writes compile-progress stub for live status
- **v57.10** UI: stat tiles show "thin input" / "single-file scope" labels on 0 values
- **v57.11** Smoke test + auto-recovery script (this file)
