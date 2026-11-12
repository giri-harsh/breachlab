#!/usr/bin/env bash
# OPTIONAL, UNUSED since ADR-0037: the Modal steps below are kept for reference only.
# Idempotent: runs every credential-gated Phase 1 step, skips steps whose env is missing, prints a table.
# Never echoes secret values. GitHub secrets/branch protection live in scripts/set_gh_secrets.sh (ADR-0022).
set -u
cd "$(dirname "$0")/.."
declare -a ROWS
have() { for v in "$@"; do [ -n "${!v:-}" ] || return 1; done; }
step() { # name, required-vars..., -- command...
  local name=$1; shift
  local need=()
  while [ "$1" != "--" ]; do need+=("$1"); shift; done; shift
  if ! have "${need[@]}"; then ROWS+=("SKIP|$name|missing: ${need[*]}"); return; fi
  if "$@" >"/tmp/finish_${name}.log" 2>&1; then ROWS+=("PASS|$name|ok"); else ROWS+=("FAIL|$name|see /tmp/finish_${name}.log"); fi
}
dvc_push() {
  uv run dvc remote modify --local origin auth basic &&
  uv run dvc remote modify --local origin user "$MLFLOW_TRACKING_USERNAME" &&
  uv run dvc remote modify --local origin password "$DAGSHUB_TOKEN" &&
  uv run dvc push
}
step check_env -- uv run python scripts/check_env.py
# Neon is migrated on a Modal CPU function (sandbox egress is HTTPS-only, ADR-0021)
step alembic MODAL_TOKEN_ID MODAL_TOKEN_SECRET NEON_DATABASE_URL -- uv run modal run src/breachlab/serve/modal_ops.py
step dvc_push DAGSHUB_TOKEN MLFLOW_TRACKING_USERNAME -- bash -c "$(declare -f dvc_push); dvc_push"
step mlflow_run MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD -- uv run python scripts/mlflow_test_run.py
step modal_cpu MODAL_TOKEN_ID MODAL_TOKEN_SECRET -- env BREACHLAB_GPU=none uv run modal run src/breachlab/serve/modal_hello.py
# GPU: Modal L4 needs a payment method (ADR-0020); Kaggle T4 kernel is the fallback
step modal_gpu MODAL_TOKEN_ID MODAL_TOKEN_SECRET MLFLOW_TRACKING_URI -- env BREACHLAB_GPU=L4 uv run modal run src/breachlab/serve/modal_hello.py
step kaggle_gpu KAGGLE_API_TOKEN -- uv run kaggle kernels push -p kaggle/hello
printf '\n%-6s %-14s %s\n' STATUS STEP DETAIL
for r in "${ROWS[@]}"; do IFS='|' read -r s n d <<<"$r"; printf '%-6s %-14s %s\n' "$s" "$n" "$d"; done
for r in "${ROWS[@]}"; do [[ $r == FAIL* ]] && exit 1; done
exit 0
