"""Merge the qwen3-embedding LoRA adapter into the base weights -> a standalone
embedding model you can run on a MAC via Ollama (no vLLM, no CUDA serving).

This is the "qwen3 quality without a big GPU" path. vLLM (CUDA-only) is the
elegant multi-LoRA serve, but you don't need it: merge once on RunPod, convert to
GGUF, and `ollama create` it on the Mac Studio alongside the base embedder.

Run on RunPod (after train_contrastive_qwen3_lora.py):
  pip install -U transformers peft accelerate
  python runpod_detector/merge_qwen3_lora.py
Then convert + serve on the Mac (see MAC_RUNPOD_PLAN.md):
  python llama.cpp/convert_hf_to_gguf.py runs/qwen3_type_merged --outfile qwen3-type.gguf
  ollama create qwen3-embed-type -f Modelfile   # Modelfile: FROM ./qwen3-type.gguf
"""
import os


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    base = os.environ.get("BASE_MODEL", "Qwen/Qwen3-Embedding-8B")
    adapter = os.environ.get("ADAPTER", "runs/qwen3_lora_unified/adapter")
    out = os.environ.get("OUT", "runs/qwen3_type_merged")

    print(f"base={base}\nadapter={adapter}\nout={out}")
    model = AutoModel.from_pretrained(base, torch_dtype=torch.float16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter)
    print("merging adapter into base weights...")
    model = model.merge_and_unload()
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    try:
        AutoTokenizer.from_pretrained(adapter, trust_remote_code=True).save_pretrained(out)
    except Exception:
        AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(out)
    print(f"merged model saved -> {out}")
    print("next: convert to GGUF (llama.cpp) then `ollama create` on the Mac.")
    print("VERIFY pooling parity: qwen3-embedding uses LAST-TOKEN pooling + the "
          "query-instruction format — embed a few atoms via the GGUF and via the "
          "trainer's encoder() and confirm cosine ~1.0 before trusting the store.")


if __name__ == "__main__":
    main()
