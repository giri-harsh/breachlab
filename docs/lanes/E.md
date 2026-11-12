# Lane E (observability) report

LANE: E (observability), branch `lane-E/observability`.

TASKS_DONE
1. `python -m breachlab.monitor.canary` (`src/breachlab/monitor/canary_cli.py`, `canary_otlp.py`; `__main__` hook in `canary.py`): cold + warm GET on `{GUARD_URL}/health` and `{BASE_URL}/api/health`, 20 probes at level 4 with the HMAC header (key = GATEWAY_SECRET_KEY, else sha256 of ADMIN_PASSWORD, same rule as the gateway), one OTLP push, exit 0/1/2. Pushes `canary_ok_total`, `canary_failures_total`, `canary_last_success_timestamp`, `canary_last_run_failures`, `canary_duration_seconds`, `guard_cold_start_seconds`, `guard_warm_latency_seconds`, `gateway_cold_start_seconds`, `gateway_warm_latency_seconds`, `guard_unavailable_total`. Standalone httpx OTLP client (no edits to telemetry.py); step summary written to `$GITHUB_STEP_SUMMARY`.
2. `.github/workflows/canary.yml`: cron `17 * * * *` + dispatch, `uv sync --no-dev` (62 runtime packages), 10 min timeout, gate step prints a notice and skips (green) if neither ADMIN_PASSWORD nor GATEWAY_SECRET_KEY exists.
3. `.github/workflows/drift.yml` + `src/breachlab/monitor/drift_job.py`: weekly (Mon 04:43 UTC) + dispatch; Neon HTTPS SQL, TF-IDF char n-grams + SVD(64) fitted on the baseline, `drift.build_report`, writes `reports/drift-YYYY-WW.md` (artifact `drift-report`), pushes `drift_*` gauges; skips on no secret or too little data (baseline < 50, window < 20).
4. OTLP ingest proof (real Grafana endpoint `https://otlp-gateway-prod-ap-south-1.grafana.net/otlp/v1/metrics`, instance 1898075): `python -m breachlab.monitor.canary --ingest-test` pushed gauge `breachlab_otlp_ingest_test` (value = unix timestamp) at 2026-10-30T05:39:58Z and earlier: `otlp ingest test: ok=True http=200 (accepted)`, both times HTTP 200. Token never printed.
5. `docs/runbook.md`: delimited "Observability" section (schedule table, how to read a red run, metric semantics, manual Grafana import steps, hand-made alert rules, no contact point by decision).
6. Dashboards: new `dashboards/ops.json` (canary OK/fail, minutes since last success, guard unavailable, guard and gateway cold vs warm latency, run duration); `drift.json` gained weekly drift panels; `alerts.json` canary rule now `sum(sum_over_time(canary_failures_total[75m]))` (the old `increase[20m]` could not fire with hourly pushes). Test asserts all dashboards parse, every metric is in `telemetry.METRICS` or the lane's lists, `${DS}` used, unique panel ids.
Live check: `python -m breachlab.monitor.canary` against https://breachlab-api.vercel.app (OTLP disabled): gateway health ok (cold 0.46 s, warm 0.43 s) but the canary session returns 500 ("cannot start canary session"), i.e. the gateway env vars (Lane D) are not set yet; `/api/health` says `guard:false`. The canary will go green once they are.

TESTS: full suite 142 passed, total coverage 97.63% (gate 80%); tests/monitor 20 tests, no network. canary.py 100%, canary_cli.py 100%, canary_otlp.py 100%, drift_job.py 99%. ruff check, ruff format --check and mypy strict (75 source files) clean.

BRANCH: lane-E/observability (pushed; commits 97d58e4, 75aa5d5, 4843070, 4549275, 1c01762 plus the report commit).

REQUESTS
- `scripts/set_gh_secrets.sh` (orchestrator-owned): add `ADMIN_PASSWORD GATEWAY_SECRET_KEY GUARD_URL GRAFANA_OTLP_ENDPOINT GRAFANA_OTLP_INSTANCE_ID GRAFANA_OTLP_TOKEN` to the `for v in ...` list (the workflows read exactly these secret names; `BASE_URL` is an optional repo variable).
- `docs/blockers.md`: one line: "GitHub Actions secrets for canary/drift | the cron workflows stay green-but-skipped until set | run scripts/set_gh_secrets.sh (after the list above is extended)".
- `CLAUDE.md` current state: add "Canary hourly (`canary.yml`) + drift weekly (`drift.yml`); OTLP ingest proven (HTTP 200); dashboards imported by hand (runbook)".
- Optional, none needed in pyproject.toml: drift.yml installs scikit-learn with `uv pip install` (it is only in the `data` group).

ADR_DRAFTS: docs/adr-drafts/E-canary-metrics-semantics.md, docs/adr-drafts/E-drift-embedding-substitute.md

BLOCKED
- Canary green run needs the gateway env vars on Vercel (Lane D) and the repository secrets (Harsh); until then local canary shows `cannot start canary session: http 500`.
- Nothing visible in Grafana can be verified without a Grafana API token or UI access (ingest accepted with HTTP 200 only).

RUNNING_JOBS: none.

NEXT: after Lane D sets the Vercel env and Harsh sets the Actions secrets, trigger `canary` once via workflow_dispatch and confirm a green run + `canary_ok_total` in Explore; import the four dashboards by hand; optionally replace `make_embedder` with a sentence-embedding model if CPU budget appears.
