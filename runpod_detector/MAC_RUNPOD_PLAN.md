# Mac-only + RunPod plan (train rented, serve free, no big GPU needed yet)

You have a Mac (Apple Silicon) and rent RunPod by the hour. The big GPU is later.
This is fine — nothing here needs a GPU you own. Split the work:

- **GPU is only for TRAINING** (a fine-tune, occasional). Rent A100 ~1hr, pull the
  artifact, release the pod. ~a few $ per run.
- **SERVING the type kNN needs no GPU** — it's numpy kNN + a small encoder on CPU.
- **Learning continues between GPU runs on CPU** (see below), so you re-fine-tune
  rarely, not constantly.

## Serving options (pick by how much quality you want now)

| Option | Where it serves | GPU to serve? | When |
|---|---|---|---|
| **bge build** (`train_contrastive_encoder_gpu.py`) | Mac CPU (sentence-transformers) | **No** | **Now — recommended default.** Train on RunPod, run on the Mac. Zero new infra. |
| **qwen3 merged → Ollama** (`merge_qwen3_lora.py` → GGUF) | Mac Studio Ollama (Metal) | **No** | If you want 8B quality before the big GPU. More setup (merge + GGUF convert), heavier RAM. |
| **qwen3 LoRA → vLLM multi-LoRA** | CUDA box (RunPod pod or your big GPU) | **Yes** | When the big GPU lands, or a persistent RunPod serve pod. The elegant one-space serve. |

**Recommendation:** start with **bge on Mac CPU**. It's the cheapest, simplest,
and (per the trade-off analysis) likely within a couple points of qwen3 on a
binary/8-way cut. Only move to qwen3 if the held-out kNN verdict says the 8B base
buys a real, worth-it gain.

## "Train over time" — how learning keeps going without constant GPU

Two clocks, and only the slow one needs the GPU:

1. **Instant (CPU, continuous):** the kNN STORE grows. Every PM correction is
   `ContrastiveTypeKNN.append(text, label)` — it influences the very next atom,
   no retrain. The existing CPU type-head / span-heads also retrain on the worker
   as the log grows (eval-gated, never-worse). So between GPU sessions the system
   keeps getting better on its own.
2. **Occasional (RunPod GPU):** re-fine-tune the encoder when the log has grown a
   lot or the store has drifted — maybe every few weeks. This bumps the ceiling;
   the store + heads handle the day-to-day.

So the cadence is: rent GPU now to fine-tune the encoder once → serve on the Mac →
let it learn on CPU → rent GPU again occasionally to lift the ceiling. You are not
paying for a GPU to sit there.

## Flow (Mac + RunPod)
```bash
# Mac: get the data, send to a rented RunPod A100
az login
KEY=$(az storage account keys list --account-name purpulsedevstg01 -g purtera-dev-rg --query "[0].value" -o tsv)
az storage blob download --account-name purpulsedevstg01 --account-key "$KEY" -c ml-artifacts -n _training_deepseek.db -f _training_deepseek.db
runpodctl send _training_deepseek.db runpod_detector/

# RunPod: train (bge default; add RUN_QWEN3_LORA=1 to also try the 8B unified space)
bash runpod_detector/run_all_gpu.sh

# Mac: pull the trained artifact, drop into the registry, serve on CPU
runpodctl receive ...        # runs/contrastive_unified/{best, store.npz}
#   -> _contrastive_type/best + _contrastive_type/store.npz
SOWSMITH_CONTRASTIVE_TYPE=1  # flip the keep-gate on once you trust the verdict
```

## How good is this, honestly (calibrated — the scripts print the truth)

Numbers are estimates until you run it; the per-epoch held-out + operating-point
lines are the real answer.

- **Baselines (measured):** frozen kNN ~0.65; classifier head ~0.82; two-model
  agreement on the keep-vs-typed boundary ~0.85 (the irreducible-ambiguity wall).
- **Expected contrastive bge (gate/unified):** ~0.82–0.88 accuracy. It should beat
  the head because it fixes the *space*, not just the classifier. It will NOT hit
  0.99 — the ~15% genuinely-ambiguous boundary is unlearnable by anyone.
- **Expected qwen3-LoRA:** maybe +2–4 pts and a bigger confident slice (8B has more
  room to reshape). Worth measuring; not guaranteed to justify the serve cost.

**The number that matters isn't accuracy — it's the GUESS-FREE OPERATING POINT:**
"type X% of atoms confidently at ≥95% precision, route the rest to the LLM." Even
at 0.85 overall accuracy, you can likely confidently auto-handle ~60–80% of atoms
at 95%+ precision and cleanly hand the ambiguous remainder to the LLM. That is the
win: most of the LLM cost/latency gone, **zero** drop in correctness, and it keeps
growing as corrections land. The system is designed so a wrong call abstains to the
LLM rather than emitting a bad label — so "how good" has a floor: it never does
worse than the LLM-only path you have today.
