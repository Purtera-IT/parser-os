# The C3 base service on a GPU (Azure Container Apps serverless T4, or any CUDA host).
# The model is not in the image: start.sh fetches c3_base.tgz (pack_live.sh, or ml/c3/pod/bundle.sh)
# from C3_BUNDLE_URL once per replica start. Ollama serves the quote heads' embedding model
# (and, if it fits and is enabled, the wording model) on localhost.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
ARG OLLAMA_VERSION=v0.6.5
RUN pip install --no-cache-dir "transformers==4.48.3" && apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tgz" | tar -xzf - -C /usr
COPY start.sh /start.sh
# A T4 has no bf16 arithmetic: read in fp16 (same weights). Wording is off by default on a T4
# (16 GB holds C3 + the embedding model, not a 14B wording model); the task line falls back.
ENV PORT=8765 SOWSMITH_C3_DEVICE=cuda SOWSMITH_C3_MIN_FREE_MB=0 C3_HALF_DTYPE=fp16 \
    OLLAMA_URL=http://127.0.0.1:11434 OLLAMA_HOST=127.0.0.1:11434 SOWSMITH_C3_QUOTE_WORDING=off \
    C3_EMBED_MODEL=qwen3-embedding:8b
EXPOSE 8765
CMD ["bash", "/start.sh"]
