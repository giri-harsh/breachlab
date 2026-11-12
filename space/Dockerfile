# Host-agnostic guard image. Build context = the output of
#   python -m breachlab.serve.space_build --stage-only <dir>
# (this folder plus the vendored runtime modules). Weights are NOT in the image: they are pulled from the private
# Hub model repo at boot with HF_TOKEN (docs/deploy-guard.md). Hugging Face Spaces run the container as uid 1000;
# the other hosts do not care.
FROM python:3.11-slim

# /cache: optional persistent volume for the downloaded weights (GUARD_CACHE_DIR=/cache)
RUN useradd -m -u 1000 user && mkdir -p /cache && chown user /cache
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONPATH=/app BREACHLAB_CONFIG_DIR=/app/configs \
    HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app

COPY --chown=user requirements.txt .
USER user
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# 7860 is the HF Spaces port; hosts that inject PORT (Cloud Run, Render) override it.
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
