# parser-os-service — production Dockerfile
#
# v45.2 ships the progress tracker.  This Dockerfile makes /v1/version honest
# by stamping the git SHA into the image, so ops can verify "is the running
# container actually the commit we think we deployed?" — the exact failure
# mode we hit during the v45.1 dev deploy (deployed d931a38 but /v1/version
# reported 1a8176c because the layer was cached).
#
# Build:
#   docker build \
#     --no-cache \
#     --build-arg GIT_SHA=$(git rev-parse HEAD) \
#     --build-arg BUILD_LABEL=v45.2 \
#     -t parserosacr.azurecr.io/parser-os-service:dev .
#
# Verify after deploy:
#   curl -sS https://<service>/v1/version | jq -r .parser_os_sha
#   # should match `git rev-parse HEAD` from the build host
#
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS runtime

# System deps: PDF parsing (poppler), OCR pre-pass (tesseract), pymupdf.
# Keep this list minimal — each adds layer weight.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject + install (cached layer when deps don't change)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -e .

# Copy app source LAST so code changes don't bust the deps layer
COPY app ./app
COPY contracts ./contracts
COPY scripts ./scripts

# ─── Git SHA stamping ──────────────────────────────────────────────────────
# These two ARGs MUST be passed at build time.  Defaults are "unknown" so the
# /v1/version endpoint surfaces the misconfiguration loudly instead of lying.
ARG GIT_SHA=unknown
ARG BUILD_LABEL=unknown

# Persist as env vars so the running container can read them at any time.
ENV PARSER_OS_SHA=$GIT_SHA
ENV PARSER_OS_BUILD_LABEL=$BUILD_LABEL

# Also write to a stamp file for belt-and-suspenders — if env is ever stripped
# by an orchestrator wrapper, _resolve_parser_os_sha() will fall back to it.
RUN echo "$GIT_SHA" > /app/.git_sha \
 && echo "$BUILD_LABEL" > /app/.build_label

# ─── Runtime ───────────────────────────────────────────────────────────────
EXPOSE 8000

# Honest healthcheck — fails the container if /health/live stops responding.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live').status==200 else 1)"

# Use uvicorn directly — bypasses gunicorn's worker-management overhead for
# a single-instance Container App.  Multi-worker is configured via Container
# App replicaCount, not gunicorn workers.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--forwarded-allow-ips=*"]
