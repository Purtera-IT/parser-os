# The C3 base service on a GPU (Azure Container Apps serverless T4, or any
# CUDA host). The model is not in the image: start.sh fetches c3_base.tgz
# (ml/c3/pod/bundle.sh) from C3_BUNDLE_URL once per replica start.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
RUN pip install --no-cache-dir "transformers==4.48.3" && apt-get update \
    && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY start.sh /start.sh
ENV PORT=8765 SOWSMITH_C3_DEVICE=cuda SOWSMITH_C3_MIN_FREE_MB=0
EXPOSE 8765
CMD ["bash", "/start.sh"]
