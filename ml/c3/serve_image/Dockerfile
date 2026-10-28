# The C3 base service on a GPU (Azure Container Apps serverless T4, or any CUDA host).
# The model is not in the image: start.sh fetches c3_base.tgz (pack_live.sh, or ml/c3/pod/bundle.sh)
# from C3_BUNDLE_URL once per replica start. Ollama serves the quote heads' embedding model
# (and, if it fits and is enabled, the wording model) on localhost.
# The encoder and tokenizer were saved by transformers 5.x: match the Mac host's stack.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# Same Ollama as the Mac host (qwen3-embedding needs >= 0.9).
ARG OLLAMA_VERSION=v0.16.2
RUN pip install --no-cache-dir "transformers==5.5.0" "tokenizers==0.22.2" "huggingface_hub==1.9.2" "safetensors==0.7.0" && apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates zstd && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" | tar --zstd -xf - -C /usr
COPY start.sh /start.sh
# A T4 has no bf16 arithmetic: read in fp16 (same weights). Wording is off by default on a T4
# (16 GB holds C3 + the embedding model, not a 14B wording model); the task line falls back.
ENV PORT=8765 SOWSMITH_C3_DEVICE=cuda SOWSMITH_C3_MIN_FREE_MB=0 C3_HALF_DTYPE=fp16 \
    OLLAMA_URL=http://127.0.0.1:11434 OLLAMA_HOST=127.0.0.1:11434 SOWSMITH_C3_QUOTE_WORDING=off \
    C3_EMBED_MODEL=qwen3-embedding:8b \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXPOSE 8765
CMD ["bash", "/start.sh"]
