#!/usr/bin/env bash
# Run LOCALLY (or anywhere `gh` is authenticated with repo+workflow scope and the env vars are exported).
# Sets every GitHub Actions secret the workflows read and protects main (requires the `checks` job).
# Never echoes values. Modal is out (ADR-0037), so no Modal secrets.
set -eu
REPO=${REPO:-giri-harsh/breachlab}
# ci cloud-smoke / promote / rollback / retrain: HF_TOKEN DAGSHUB_TOKEN MLFLOW_* KAGGLE_API_TOKEN
# canary / drift (Lane E): ADMIN_PASSWORD GATEWAY_SECRET_KEY GUARD_URL GRAFANA_OTLP_* NEON_DATABASE_URL
for v in HF_TOKEN DAGSHUB_TOKEN MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD \
  KAGGLE_API_TOKEN NEON_DATABASE_URL GROQ_API_KEY GEMINI_API_KEY \
  ADMIN_PASSWORD GATEWAY_SECRET_KEY GUARD_URL GRAFANA_OTLP_ENDPOINT GRAFANA_OTLP_INSTANCE_ID GRAFANA_OTLP_TOKEN; do
  if [ -n "${!v:-}" ]; then
    printf '%s' "${!v}" | gh secret set "$v" --repo "$REPO" >/dev/null && echo "set $v"
  else
    echo "skip $v (unset)"
  fi
done
# Optional repo variable: where canary/keep-warm probe the gateway (default in the workflows).
if [ -n "${BASE_URL:-}" ]; then gh variable set BASE_URL --repo "$REPO" --body "$BASE_URL" && echo "set variable BASE_URL"; fi
gh api -X PUT "repos/$REPO/branches/main/protection" -H "Accept: application/vnd.github+json" --input - <<'JSON'
{"required_status_checks":{"strict":false,"contexts":["checks"]},"enforce_admins":false,"required_pull_request_reviews":null,"restrictions":null}
JSON
echo "branch protection set on main"
