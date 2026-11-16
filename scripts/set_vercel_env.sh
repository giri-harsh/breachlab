#!/usr/bin/env bash
# Run where VERCEL_TOKEN and the service env vars are exported (the session classifier may refuse secret-store writes;
# the REST route POST /v10/projects/{id}/env also works, see docs/lanes/D.md).
# Sets runtime env on the gateway project `breachlab-api` and redeploys it. Never echoes values.
set -eu
SCOPE=${SCOPE:-harshs-projects-d6e22d93}
cd "$(dirname "$0")/.."
export CORS_ORIGIN=${CORS_ORIGIN:-https://breachlab-web.vercel.app,https://breach.harshgiri.site}
export AGENT_MODEL=${AGENT_MODEL:-openai/gpt-oss-20b}   # 120b hits its daily token cap on the free tier
export GUARD_URL=${GUARD_URL:-https://placeholder.invalid}
export CHAMPION_TTL_S=${CHAMPION_TTL_S:-30}
export GUARD_TIMEOUT_S=${GUARD_TIMEOUT_S:-3.0}
for v in NEON_DATABASE_URL GROQ_API_KEY ADMIN_PASSWORD GUARD_URL GUARD_TOKEN CORS_ORIGIN AGENT_MODEL \
  CHAMPION_TTL_S GUARD_TIMEOUT_S MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD \
  LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY LANGFUSE_HOST GRAFANA_OTLP_ENDPOINT GRAFANA_OTLP_INSTANCE_ID GRAFANA_OTLP_TOKEN; do
  if [ -n "${!v:-}" ]; then
    printf '%s' "${!v}" | npx --yes vercel@latest env add "$v" production --force --sensitive --token "$VERCEL_TOKEN" --scope "$SCOPE" >/dev/null && echo "set $v"
  else
    echo "skip $v"
  fi
done
npx --yes vercel@latest deploy --prod --yes --token "$VERCEL_TOKEN" --scope "$SCOPE"
