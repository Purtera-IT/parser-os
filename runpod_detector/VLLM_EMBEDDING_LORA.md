# Unified embedding space via qwen3-embedding LoRA + vLLM

The "one space for everything" path you flagged. Straight assessment: **it's
viable and elegant — the catch is purely infra (vLLM is CUDA-only), not the ML.**

## The idea
Today there are two embedding spaces:
- **qwen3-embedding:8b** (Ollama, Mac Studio) → the general decide()/feedback store.
- a **separate bge** fine-tuned for the type/facet kNN (the default contrastive build).

Unifying means LoRA-fine-tuning **qwen3-embedding itself** on the contrastive
objective, so the type kNN lives in the *same* 4096-d space as everything else.

## Why Ollama can't, but vLLM can
- Ollama has no custom-embedding-LoRA serving. Dead end.
- **vLLM serves embeddings (`--task embed`) AND multiplexes LoRA adapters
  (`--enable-lora`)** from one process. Qwen3-Embedding is a Qwen3 decoder with
  last-token pooling, so a LoRA adapter on its projections is a first-class case.

The elegant part — **multi-LoRA from one served model** respects the pinned
"never swap the embed model" rule:
```
vllm serve Qwen/Qwen3-Embedding-8B --task embed --enable-lora \
  --lora-modules type=runs/qwen3_lora_unified/adapter
```
- request with **no adapter** → base embedding → the existing decide() store,
  byte-for-byte unchanged (pinned thresholds/heads stay valid).
- request with **`model: "type"`** → base+adapter → the re-sorted type/facet space.

One model in memory (8B), the adapter is ~tens of MB, switched per-request.

## The real trade-offs (honest)
| | |
|---|---|
| **CUDA-only** | vLLM does not run on Apple Silicon. The Mac Studio (Metal) **cannot** serve this. Prod embedding must move to a CUDA box — a persistent GPU (RunPod/Azure), not just a training burn. This is the one hard cost. |
| **Numerical parity** | Before trusting base-vLLM for the *existing* store, verify vLLM base embeddings match Ollama's for the same model/dtype (cosine ≈ 1.0 on a sample). They should (same weights), but the pinned store was built on Ollama — re-validate, or re-embed the store on vLLM once and pin to that. |
| **Training weight** | This is a custom HF + PEFT loop on an 8B model, heavier than the bge sentence-transformers fit. A100 80GB fits it (bf16); 40GB needs `QLORA=1` + grad-checkpointing. |
| **Serving cost** | An always-on 8B on a CUDA GPU vs bge on CPU. bge is far cheaper to serve. |

## Decision rule
Run **both** trainers on the A100 and compare held-out kNN:
- `train_contrastive_encoder_gpu.py`   (bge, separate space — cheap to serve)
- `train_contrastive_qwen3_lora.py`    (qwen3 LoRA, unified space — needs vLLM/CUDA)

Then:
- **bge unified ≥ ~0.82 and within a couple points of qwen3** → ship bge. Cheapest,
  CPU-served, no infra change. The unified space is nice-to-have, not worth a GPU.
- **qwen3 LoRA clearly higher** (the 8B base has more room to reshape) → the gain
  justifies standing up vLLM-on-CUDA for prod embedding; serve base + `type`
  adapter, retire the bge model, one space everywhere.

The runtime (`app/core/contrastive_type_knn.py`) is **encoder-agnostic**: it takes
an `embed_fn`. bge → loads the saved sentence-transformers encoder. qwen3 → pass an
`embed_fn` that POSTs to the vLLM `/v1/embeddings` endpoint with `model: "type"`.
The kNN store + abstain logic are identical, so whichever wins drops in without a
runtime rewrite.

## Serving the qwen3 path (if it wins)
```bash
pip install "vllm>=0.6.3"
vllm serve Qwen/Qwen3-Embedding-8B --task embed --enable-lora \
  --lora-modules type=runs/qwen3_lora_unified/adapter --max-lora-rank 16
# worker embed_fn: POST /v1/embeddings  {"model":"type","input":[...texts...]}
# store: runs/qwen3_lora_unified/store.npz  (re-embed if you bump the adapter)
```
Standing constraints unchanged: commit CODE only (never the adapter/.npz/.db);
guess-free abstain + eval-gate before any cutover.
