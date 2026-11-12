#!/usr/bin/env bash
# One command once a guard host exists: GUARD_URL=https://<host> VERCEL_TOKEN=... scripts/set_guard_url.sh [--dry-run|--no-live|--ref main]
# Validates /health and /score, sets GUARD_URL on breachlab-api, flips the web "guard offline" flag, redeploys both, runs the live spec.
set -eu
cd "$(dirname "$0")/.."
exec python3 scripts/set_guard_url.py "$@"
