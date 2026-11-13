# Runbook

## Current state (read first)
- **Live, all five levels** (ADR-0082): web https://breachlab-web.vercel.app, gateway https://breachlab-api.vercel.app, guard https://breachlab-guard.vercel.app. The live Playwright spec L1-L5 passes on production with no skips.
- **The guard is a Vercel Hobby function** (project `breachlab-guard`, 1 vCPU, 2 GB, source in `vercel-guard/`): the int4 `q4b32` ONNX graph (433 MB, `champion/guard-q4b32.onnx` on the private repo `Pseudo-Harsh/breachlab-guard-v1`) fetched into RAM at cold start with `HF_TOKEN`, with its own decision threshold 0.23195 (`champion/calibration-q4b32.json`). `POST /score` needs `Authorization: Bearer <GUARD_TOKEN>` (401 without); `GET /health` and `GET /vercel-info` are open.
- **Measured latency** (ADR-0082/0083, real function, 500 requests per level, client in the sandbox): concurrency 1 round trip p50 842 / p95 3556 ms (model time p50 720 / p95 3468 ms); concurrency 4 round trip p50 3664 ms (the single vCPU queues requests). Cold first request about 13 s. **The 300 ms target is NOT met.** The v1 gate uses `p95_latency_ms_max` 4161 ms (measured model-time p95 x 1.2), which means "within the relaxed v1 limit", not "meets 300 ms".
- **Cold start is fail-closed.** After more than about 2 minutes idle the instance may be gone (it was gone at 10 minutes; the span in between is unmeasured, [`docs/keep-warm.md`](keep-warm.md)). The first request boots it (about 13 s); the gateway waits `GUARD_TIMEOUT_S=8`, so that player gets `503 guard waking up` (ADR-0050) and the next request works. This is a known limit, not an outage to fix in code. Nothing keeps the guard warm today.
- **One graph only.** Aliases `base` and `challenger` fall back to the champion, so level 3 uses the same guard graph as level 4 (the untuned base guard of ADR-0045 is not served).
- **Open quality item** (ADR-0083, `docs/blockers.md`): served int4 gold FPR 2.08% vs the 2.0% bar; the bar was not relaxed.
- `/api/health` shows `guard` (= `GUARD_URL` is set) and `guard_ok` (a cached 60 s probe of the guard's `/health` with a 1 s timeout; the web nav pill reads it). A cold or slow guard can show `guard_ok: false` while the guard is fine (seen on 2026-11-01 about 08:40 UTC with the guard answering 200 directly; not diagnosed). Trust `GET https://breachlab-guard.vercel.app/health` and a real level 4 message over that flag.
- **Cold-start handling is live (S14, ADR-0085; deployed 2026-11-11).** The web app fires `GET /api/warm` on page load and session start; the gateway uses a 20 s cold timeout and answers HTTP 202 "warming" (UI retries by itself) before failing closed with 503. Measured first message after 13 minutes idle: 19.65 s before, 17.81 s without pre-warm, 8.4 s with pre-warm and 8 s of typing (one run each). A player who sends at once still waits about 18 s.
- **Deploy rule (ADR-0089).** At most 5 Vercel deploys per day, and only when there is no other way to verify a change (try a local test, a mocked Playwright run or a read-only check first). Pushing a branch builds previews on both projects, so do not push only to refresh a heartbeat. Cheapest durable fix (needs Harsh): an Ignored Build Step on `breachlab-api` and `breachlab-web` that builds only `main`: `[ "$VERCEL_GIT_COMMIT_REF" = "main" ] && exit 1 || exit 0`.
- **Vercel deploy quota:** it had reset by 2026-11-11 (deploys then succeeded). Redeploy helper: `Vercel(token).redeploy(name, project_id, "main")` in `scripts/set_guard_url.py`; check `/openapi.json` for `/api/warm` (does not touch the guard); `scripts/measure_cold_message.py` needs 12+ idle minutes with no guard request first.
- **v2, not in v1:** a separate weaker guard for level 3 (ADR-0086) and scoring the 300-row opener set (ADR-0087). The Kaggle kernels `giriharsh/b3b-val` and `giriharsh/b3b-gold` hold the base-model int4 logits for the first of these.
- Budget is $0 (ADR-0037). Modal is out. Everything below uses free tiers.

## Where things run
| Part | Host | Notes |
|---|---|---|
| Frontend | Vercel project `breachlab-web` | `NEXT_PUBLIC_API_BASE` in `web/.env.production` |
| Gateway | Vercel project `breachlab-api` | runtime env set through the Vercel REST API (ADR-0056); `scripts/set_vercel_env.sh` is the CLI equivalent. Guard env: `GUARD_URL`, `GUARD_TOKEN`, `GUARD_TIMEOUT_S=8` |
| Guard | Vercel project `breachlab-guard` (https://breachlab-guard.vercel.app), `vercel-guard/` | env `HF_TOKEN` and `GUARD_TOKEN` (write-only), `GUARD_MODEL_REPO`, `GUARD_ONNX_FILE`, `GUARD_CALIBRATION_FILE`, `GUARD_THREADS`; deploy with `scripts/deploy_vercel_guard.py`. Other hosts (lower latency, second graph): [`docs/deploy-guard.md`](deploy-guard.md) |
| DB | Neon Postgres (Alembic migrations through CI or a local run; sandbox egress is HTTPS-only, use the HTTPS SQL endpoint there) | |
| Training | Kaggle T4 kernels (`train/kaggle_job.py`, slugs are lane-prefixed, ADR-0059) | weights never enter the dev container (ADR-0040) |
| Registry | MLflow on DagsHub, model `breachlab-guard`, aliases champion/challenger/previous | `base` is not an alias, it is the untuned Qwen3-0.6B (not served on the Vercel host) |
| Data / models | DVC on DagsHub; private HF repos `Pseudo-Harsh/breachlab-data` and `Pseudo-Harsh/breachlab-guard-v1` | cards: `uv run python scripts/make_hf_cards.py` (README.md only, repos stay private) |

## Redeploying the guard and re-pointing the API
Secrets only come from the environment; no step prints them.
1. **Redeploy the guard function**: `VERCEL_TOKEN=... HF_TOKEN=... GUARD_TOKEN=... uv run python scripts/deploy_vercel_guard.py` (add `--onnx-file guard-q4b32.onnx` to choose the graph; `--no-deploy` only upserts the project and env). It creates the project if missing with Deployment Protection off (the gateway calls it server to server; `/score` is guarded by the bearer token), upserts env (`HF_TOKEN`, `GUARD_TOKEN` as write-only; only variables present in your environment are touched), stages `vercel-guard/` without git metadata (ADR-0055) and deploys to production. The new instance boots cold (about 13 s on the first request).
2. **Check the new deployment**: `curl https://breachlab-guard.vercel.app/health` (200 and `"champion"` loaded; the first call can take about 13 s), then `curl https://breachlab-guard.vercel.app/vercel-info` (`boot_error` must be null).
3. **Re-point the gateway** (needed when the guard URL or token changed, harmless otherwise): `GUARD_URL=https://breachlab-guard.vercel.app GUARD_TOKEN=... VERCEL_TOKEN=... uv run python scripts/set_guard_url.py` (`--dry-run` stops after validation; `--no-live` skips the Playwright spec; `scripts/set_guard_url.sh` is the shell variant). It checks `GET /health` and a real `POST /score`, sets `GUARD_URL` (and `GUARD_TOKEN`, write-only) on `breachlab-api`, sets `NEXT_PUBLIC_GUARD_OFFLINE=0` on `breachlab-web`, redeploys both from `main` and runs the live L1-L5 spec. Nothing is changed before the host validates. It does not touch `GUARD_TIMEOUT_S`, which stays 8 unless you change it (`scripts/set_vercel_env.sh`).
4. **After a change of graph or threshold**: re-run the leak suite and the benchmark (below), then update `configs/gate.yaml` `p95_latency_ms_max` (measured model-time p95 x 1.2), `scripts/make_report.py`, `uv run python scripts/readme_numbers.py fill` and `uv run python scripts/check_readme_numbers.py`, in a new ADR.

## Rotating GUARD_TOKEN
The token must be identical on the guard (`breachlab-guard`) and the gateway (`breachlab-api`). A mismatch makes every level 3-5 message fail closed (the guard answers 401, the gateway answers 503), so do both steps back to back and test.
1. Generate a new value locally (for example `export GUARD_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"`); do not print or commit it.
2. Guard: run `VERCEL_TOKEN=... HF_TOKEN=... uv run python scripts/deploy_vercel_guard.py` with the new `GUARD_TOKEN` exported (this sets it on `breachlab-guard` and redeploys the guard). By hand: change the env var `GUARD_TOKEN` on the Vercel project `breachlab-guard` (Production) and redeploy it.
3. Gateway: run `GUARD_URL=https://breachlab-guard.vercel.app VERCEL_TOKEN=... uv run python scripts/set_guard_url.py` with the same `GUARD_TOKEN` exported (this sets it on `breachlab-api`, redeploys `breachlab-api` and `breachlab-web`, and runs the live spec). By hand: change `GUARD_TOKEN` on `breachlab-api` and redeploy it.
4. Verify: `curl -s -o /dev/null -w '%{http_code}\n' -X POST https://breachlab-guard.vercel.app/score -H 'content-type: application/json' -d '{"text":"hi","kind":"input","alias":"champion"}'` must print 401 (no token), and a level 4 message in the game must work. Update the GitHub Actions secret `GUARD_TOKEN` too if it is set there (`scripts/set_gh_secrets.sh`; the canary calls only the open `/health`, so it is not affected). Local copies used for the leak suite and benchmark need the new value.

## Common operations
- **Health**: `curl https://breachlab-api.vercel.app/api/health` (200; see the `guard` / `guard_ok` caveat above) and `curl https://breachlab-guard.vercel.app/health`.
- **Warm the guard before a demo**: `curl https://breachlab-guard.vercel.app/health` 15-30 s before, and again after more than 2 minutes of silence ([`docs/demo-script.md`](demo-script.md)). One request is enough; do not loop, the function has one vCPU.
- **Play-test**: https://breachlab-web.vercel.app, levels 1-5.
- **Live Playwright spec (levels 1-5 against production)**: `cd web && E2E_BASE_URL=https://breachlab-web.vercel.app E2E_LIVE=1 ADMIN_PASSWORD=... PW_CHROMIUM=/opt/pw-browsers/chromium npx playwright test live` (`PW_CHROMIUM` only when the sandbox needs that Chromium). Warm the guard first: with a cold guard a level 3-5 test fails with "guard offline" (loudly, by design) and passes on a re-run. Expect 5 passed. Keep `E2E_EXPECT_GUARD_OFFLINE` unset.
- **Leak suite L1-L5**: `GROQ_API_KEY=... GUARD_URL=https://breachlab-guard.vercel.app GUARD_TOKEN=... uv run python scripts/run_leak_suite.py [levels...]` (about 8 minutes for L3-L5; in-process `GameService`, canary sessions on an in-memory DB, Groq `gpt-oss-20b` at temperature 0, `GUARD_TIMEOUT_S` default 8). It writes `data/eval/leak_suite.json` and resumes per level and per attack (`data/eval/leak_suite_progress.json`): a level with a stored result is skipped, so remove that level from the JSON to re-run it. Failures (503 guard waking up, vault busy) are retried with backoff and never counted as attacks. Remember what a zero means (ADR-0083): at L3-L5 the input guard blocked all 216 attacks, so the vault and the L5 output filter were not exercised.
- **Re-measure guard latency**: `GUARD_URL=https://breachlab-guard.vercel.app GUARD_TOKEN=... uv run python scripts/bench_vercel_guard.py --gold <gold_test.jsonl> --out bench.json` (500 requests at concurrency 1 and 4, then cold gaps; about 20 minutes and it loads the single vCPU, so never while players or the canary are active). Earlier results: `docs/lanes/B3-vercel-bench-*.json`.
- **Roll back a bad model**: run workflow `rollback` (or `python -m breachlab.flywheel.registry rollback`). Alias move takes 2-6 s on DagsHub (3 cycles on a test model, `docs/lanes/P-rollback-timing.json`); the gateway re-reads the alias every 30 s, so a rollback is live in about 36 s worst case. Note the Vercel guard serves the file named in `GUARD_ONNX_FILE`, not the registry alias: a different graph needs `deploy_vercel_guard.py --onnx-file ...`.
- **Promote**: workflow `promote` (runs `scripts/eval_gate.py` first).
- **Review queue**: `/admin/review` (ADMIN_PASSWORD); keys s/i/j/e/k/u. 300 gold rows plus 93 judge disagreements wait in session `gold-review-v1` for a human pass.
- **Canary**: see Observability below.
- **Self-play traffic**: `uv run python -m breachlab.flywheel.selfplay --base-url https://breachlab-api.vercel.app --sessions 100 --max-attempts-per-level 8`; rows are labeled `source=selfplay` (nicknames `selfplay-...`). Levels 3-5 need the warm guard and the single vCPU serializes, so keep runs small and pace them. It shares the Groq quota with the vault agent, so stop on repeated 429. Label with `python -m breachlab.flywheel.judge`, build data with `python -m breachlab.flywheel.build_dataset`.
- **Cards**: `uv run python scripts/make_hf_cards.py --dry-run` writes the HF cards to `reports/hf-cards/`; without the flag it pushes README.md to the two private HF repos. It never changes visibility.
- **Rotate another secret**: change it in the provider, re-run `scripts/set_vercel_env.sh` / `scripts/set_gh_secrets.sh`. For `GUARD_TOKEN` see the section above.
- **Budget**: caps in `configs/budget.yaml` ($25 per job, $150 per month; actual spend is $0). Training runs log GPU-seconds and USD to MLflow.
- **Link check**: `uv run python scripts/check_links.py` (README.md and docs/*.md: relative links offline, https links with HEAD; private GitHub / HF links are reported as "private, not checked"). `uv run python scripts/link_check.py` is the older check of the README and web app links (WARN on private hosts until the repos are public, `LINKCHECK_STRICT=1` after release).

## Failure modes
- Guard cold (idle more than about 2 minutes): the first level 3-5 message returns 503 "guard waking up" (about 13 s boot vs the 8 s gateway wait); the UI retries once and may show "guard offline". `curl https://breachlab-guard.vercel.app/health` (takes about 13 s), then retry. Not a regression.
- Guard unreachable, 401 or crashing: levels 3-5 return the same 503 and never fail open. Check `GET https://breachlab-guard.vercel.app/health`, `/vercel-info` (`boot_error`), the Vercel logs of `breachlab-guard`, a `GUARD_TOKEN` mismatch (see rotation above), and `guard_unavailable_total`.
- Several players at once: the single vCPU serializes requests (concurrency 4 gave a median round trip of 3.7 s); some may exceed 8 s and see 503. Documented limit.
- Groq 429 on the vault agent: the agent honours `retry-after`, then falls back along `AGENT_MODELS` (default gpt-oss-20b, qwen3.8-27b, gpt-oss-120b; ADR-0052); if the whole chain is exhausted the game answers 503 "vault busy". Self-play and the judge share these free quotas.
- Neon unreachable from a sandbox (TCP blocked): use the HTTPS SQL endpoint or run in CI.
- Kaggle kernel finished but the status still says ERROR: the status call can lag a version; wait for RUNNING then terminal.
- Never download kernel outputs without `--file-pattern '.*\.(log|json|npy)$'` (weights must not enter the dev container). The only weight movement allowed is the transit-only HF relay (ADR-0040).
- Vercel marks a CLI deploy BLOCKED: the commit author is not a team member; deploy from a git-less copy (ADR-0055). `deploy_vercel_guard.py` already does that.

<!-- BEGIN observability (lane E; ADR-0041, ADR-0042, ADR draft E2-canary-degraded-mode) -->
## Observability

### What runs on a schedule
| Workflow | When | What |
|---|---|---|
| `canary` | hourly at :17 (+ manual `workflow_dispatch`) | warm-up + measured GET on `{BASE_URL}/api/health` (and `{GUARD_URL}/health` when `GUARD_URL` is a real URL), then **tier 1**: 6 probes at levels 1-2 (no guard needed), then one benign level-4 probe that decides **tier 2**: the 20 PRD probes at level 4. HMAC `X-Canary` header throughout; one OTLP push to Grafana Cloud at the end |
| `drift` | Mondays 04:43 UTC (+ manual) | last 7 days of non-canary attempts from Neon vs. the older baseline; writes `reports/drift-YYYY-WW.md` (workflow artifact `drift-report`) and pushes `drift_*` gauges |

The guard probe is not a keep-warm: the Vercel guard function is gone after about 10 minutes idle and the canary runs hourly, so most runs find it cold and pay the roughly 13 s boot themselves (what is measured and what is not: [`docs/keep-warm.md`](keep-warm.md)). Secrets come from repository secrets (`ADMIN_PASSWORD` or `GATEWAY_SECRET_KEY`, `GUARD_URL`, `GRAFANA_OTLP_*`, `NEON_DATABASE_URL`; set with `scripts/set_gh_secrets.sh`); `BASE_URL` may be a repository variable (default `https://breachlab-api.vercel.app`). With no signing secret the run stays green and the `check secrets` step prints a notice and a job-summary table naming exactly which secrets are missing (required: `ADMIN_PASSWORD` or `GATEWAY_SECRET_KEY`; optional: `GUARD_URL`, `GRAFANA_OTLP_ENDPOINT`, `GRAFANA_OTLP_INSTANCE_ID`, `GRAFANA_OTLP_TOKEN`) plus the one fix: export the values locally and run `scripts/set_gh_secrets.sh` (only names are ever printed, never values; a skipped run bills about one minute); with no `GRAFANA_OTLP_*` it still runs the checks and just skips the push. Local run: `uv run python -m breachlab.monitor.canary` (exit 0 ok, 1 failed, 2 not configured); `--ingest-test` only pushes the `breachlab_otlp_ingest_test` gauge and prints the HTTP status.

### Degraded versus red (ADR draft E2-canary-degraded-mode)
The canary has three outcomes. Read the first line of the job summary:

| Summary heading | Run colour | Meaning | Metrics |
|---|---|---|---|
| `Canary: PASS` | green | tier 1 and tier 2 passed, guard up | `guard_up=1`, `canary_ok_total=1`, `canary_degraded_total=0` |
| `Canary: DEGRADED (guard offline, tier 1 passed)` + a `canary degraded` notice | green | the gateway, the vault agent and the regex wall (levels 1-2) are healthy; the guard host is missing or unreachable, so levels 3-5 return 503 by design (ADR-0063). Tier 2 was skipped | `guard_up=0`, `canary_degraded_total=1`, `canary_failures_total=0`, `canary_ok_total=1` |
| `Canary: FAIL` | red | a real regression (list under "Failures") | `canary_failures_total=N`, `canary_ok_total=0` |
| `Canary: SKIPPED (no signing secret)` | green | no Actions secret yet; nothing was probed and nothing was pushed | none |

How "guard offline" is decided: after tier 1 the canary sends one benign message at level 4. A gateway answer of `503` whose detail contains "guard" (`guard waking up, retry in a few seconds`) means offline. The same applies when `/api/health` reports `guard: false`. A 503 "the vault is busy" is Groq rate-limiting, is retried twice (4 s, 8 s) and is never mistaken for a guard outage. Any other level-4 error (500, 401, ...) is a real failure.

Offline is degraded **only while no guard host is expected**: `GUARD_URL` is unset or the placeholder (`https://placeholder.invalid`, anything containing `placeholder`, `.invalid` or `example.`). It is a **red failure** when `GUARD_URL` is a real URL (the host exists, so being down is a regression: either `guard health: ...` or `guard offline (...) but a guard is expected`), or when the repository variable `CANARY_REQUIRE_GUARD` is `1`.

**Guard host today**: the guard is live on https://breachlab-guard.vercel.app (ADR-0082), so once the Actions secrets exist the run goes through tier 2 and the heading becomes `PASS`. Recommended once: set the Actions secret `GUARD_URL` to https://breachlab-guard.vercel.app (`scripts/set_gh_secrets.sh`; this also adds the cold/warm guard latency panels) and optionally the repository variable `CANARY_REQUIRE_GUARD=1` so a later guard outage can never be hidden as "degraded". If tier 2 then fails with `attack passed the guard` or `benign message blocked`, that is a model regression: see below.

### Reading a red canary run
Open the run, then the job summary. The table shows each target's status plus cold-start and warm latency; "Failures" lists each broken check:
- `guard health: http 503, then 503` : the Vercel guard function is down or still booting past the probe's retry (a normal cold start takes about 13 s and stays green with a warning). Check the Vercel logs of `breachlab-guard` and `/vercel-info` (`boot_error`), redeploy with `scripts/deploy_vercel_guard.py`; levels 3-5 fail closed (503) meanwhile. `guard_unavailable_total` was pushed.
- `gateway health: ...` : Vercel `breachlab-api` is down or mis-deployed (probes are skipped). `curl /api/health`; look at the Vercel deployment logs.
- `http 500: '<session>'` : the canary session could not start (usually DB env vars on the gateway or a wrong signing key: the key must equal the gateway's `GATEWAY_SECRET_KEY`, or the sha256 derived from `ADMIN_PASSWORD`).
- `attack passed the guard` / `attack leaked the secret` : the guard regressed or is mis-thresholded; roll back with the `rollback` workflow.
- `benign message blocked` : false positive on a canary probe; also a candidate for rollback.
- `tier1 L1 ...` / `tier1 L2 ...` : a level 1-2 probe failed with the guard irrelevant. `tier1 L2 attack passed the guard` means the regex wall (`serve/baselines.py`) or the gateway changed; `tier1 L2 benign message blocked` is a regex false positive; `http 5xx` is the gateway or the vault agent. The tier-1 probe set is pinned to the regex by a unit test, so edit `TIER1_PLAN` in `monitor/canary.py` if the regex changes on purpose.
- `guard probe at level 4: http 500` : the gateway failed in a way that is not the documented guard 503; tier 2 was skipped.
- `http 503` on many probes : guard or Groq unavailable at the time (rate limit); re-run via `workflow_dispatch` before escalating.
A green run with a `::warning::OTLP push failed` annotation means the checks passed but Grafana rejected or missed the metrics (check the token/endpoint secrets).

### Metrics pushed by the canary (service `breachlab-canary`)
`canary_ok_total` (1 or 0 per run), `canary_failures_total` (failed checks in the run), `canary_last_success_timestamp`, `canary_last_run_failures`, `canary_duration_seconds`, `guard_cold_start_seconds`, `guard_warm_latency_seconds`, `gateway_cold_start_seconds`, `gateway_warm_latency_seconds`, `guard_unavailable_total` (the expected guard host failed its health check), `guard_up` (gauge, 1 = level-4 probe answered, 0 = gateway reported the guard offline; absent when the probe could not run) and `canary_degraded_total` (1 on degraded runs). Degraded runs push `canary_failures_total=0` and leave `guard_unavailable_total` at 0: the known offline state is never counted as a failure. The process is short-lived, so counters are per-run deltas: query them with `sum_over_time` (not `increase`) and gauges with `last_over_time(...[2h])`, because an hourly sample is stale after 5 minutes. Weekly drift: `drift_share_outside_known`, `drift_new_attempts`, `drift_flagged`.

### Grafana: manual dashboard import (no API token, ADR-0042)
1. Grafana Cloud stack > Dashboards > New > Import > Upload dashboard JSON file.
2. Upload each of `dashboards/live-game.json`, `model.json`, `drift.json`, `ops.json` in turn.
3. When prompted, choose the stack's Prometheus data source (named `grafanacloud-<stack>-prom`); each dashboard has a `DS` data-source variable at the top, set it to the same source if it shows "No data".
4. Metrics land under their OTLP names (`requests_total`, `canary_ok_total`, ...). If panels stay empty, run `uv run python -m breachlab.monitor.canary --ingest-test` and look for `breachlab_otlp_ingest_test` in Explore.

### Alerts: none by decision
No contact point is configured (ADR-0025/0042): nothing pages anyone and the red GitHub Actions run (plus GitHub's own failure e-mail) is the alert. `dashboards/alerts.json` keeps the D6 rule definitions as code. To add them by hand: Alerting > Alert rules > New alert rule > pick the Prometheus source, paste the `expr` of a rule from `alerts.json` into query A, add a threshold expression with the same evaluator, set the evaluation interval and `for` from the file. The canary rule uses `sum(sum_over_time(canary_failures_total[75m])) > 0`. The ops dashboard (`ops.json`) has `guard_up` and degraded-run panels. Optional hand-made rule for a guard that was up and disappeared (it does not fire in the never-provisioned state): query A `(max(max_over_time(guard_up[7d])) > 0) and (max(max_over_time(guard_up[3h])) == 0)`, threshold `> 0`, relative range 7 d, `for` 0s, no data = OK. `scripts/provision_grafana.py` can do all of this at once if a service-account token (Editor) is ever created.
<!-- END observability -->

## L3 base guard host and the opener limitation (ADR-0090)
- The weak L3 guard (zero-shot base, int4) is measured and on the Hub, but not deployed. To deploy it on a day with at least 2 deploys left: `VERCEL_TOKEN=... HF_TOKEN=... GUARD_TOKEN=<same value as the champion host> uv run python scripts/deploy_vercel_guard.py --name breachlab-guard-base --alias base --onnx-file guard-q4b32.onnx --calibration-file calibration-q4b32.json`; set `GUARD_BASE_URL=the `breachlab-guard-base` Vercel project (retired, ADR-0090)` on `breachlab-api`; redeploy the API; check `/api/warm` lists both hosts; run the leak suite at level 3 and the live Playwright spec once; update `reports/eval-v1.md`, `/under-the-hood` and the README statement that L3 uses the champion.
- Players who write Hindi or Hinglish, or send very short openers, are often blocked at levels 3-5 (52% of 300 short benign openers, 94-96% of the Hindi ones). This is a training-data gap (ADR-0090), not a threshold problem; do not retune the threshold on prod prompts.


## Database migrations (ADR-0095)
Neon is reached over HTTPS only from the dev container, so `alembic upgrade head` (Postgres TCP) cannot run there. Procedure used for 0002 (2026-11-12):
1. **Backup first**: dump every public table through the HTTPS SQL endpoint (`https://<neon host>/sql`, header `Neon-Connection-String`) to `backups/neon-<date>/<table>.jsonl` plus `schema.json`; `dvc add backups/neon-<date>` and `dvc push` (the folder is git-ignored: it holds player nicknames and prompts). Compare row counts with `select count(*)`. A Neon branch backup needs a Neon API key, which this project does not have.
2. **Apply** the same additive statements the Alembic revision holds, in one request, and bump the version table: `alter table levels add column if not exists ...; update alembic_version set version_num='0002' where version_num='0001';` From a host with TCP, `uv run alembic upgrade head` does the same.
3. **Verify**: `select version_num from alembic_version` (this is `alembic current`), `select count(*), sum(points), sum(hints_used), count(skipped_at) from levels`, and `GET /api/leaderboard` returns 200.
4. **Rollback**: `alembic downgrade -1` from a TCP host, or over HTTPS `alter table levels drop column skipped_at, drop column points, drop column hints_used; update alembic_version set version_num='0001' where version_num='0002';` (the old API code ignores the columns, so rolling the schema back is only needed if the new columns misbehave). Full restore: replay the `backups/` JSONL into the tables (`dvc pull backups/neon-<date>.dvc`).
Order rule for the next migration: apply the additive migration BEFORE merging the code that needs it (the 2026-11-12 merge went first and `/api/leaderboard` returned 500 for about an hour).

## Guard status: ok / warming / down (ADR-0098)
`GET /api/health` returns `guard_state`: `ok` (the guard answered `/health` within 2 s), `warming` (it did not: a cold serverless guard needs about 10-15 s to boot; a non-blocking warm with a 20 s timeout is fired), `down` (last successful contact older than 10 min or never, AND a warm attempt failed). `guard_ok` is true only for `ok`. The web polls every 2 s for up to 30 s while `warming`, shows "BOSS WAKING UP… (~10 s)" for bosses 3-5 and opens them by itself; "offline" (with RETRY) appears only for `down`. Pre-warm: `/api/warm` on home and play load, and on hover/focus of boss 3-5 tiles/tabs (one call per 8 s). There is no build-time flag.
Check by hand after the guard has been idle 15+ minutes (do not hit the API meanwhile, `/api/health` itself pings the guard): `curl -s $API/api/health` (expect `warming` the first time), poll it every 2 s until `ok`, and load the home page in a browser to confirm the nav pill says "guard waking up" and boss 3 shows BOSS WAKING UP… and never "offline".

## Guard status: ok / warming / down (ADR-0098)
`GET /api/health` returns `guard_state`: `ok` (the guard answered `/health` within 2 s), `warming` (it did not: a cold serverless guard needs about 10-15 s to boot; a non-blocking warm with a 20 s timeout is fired), `down` (last successful contact older than 10 min or never, AND a warm attempt failed). `guard_ok` is true only for `ok`. The web polls every 2 s for up to 30 s while `warming`, shows "BOSS WAKING UP… (~10 s)" for bosses 3-5 and opens them by itself; "offline" (with RETRY) appears only for `down`, or when the gateway itself never answered within the 30 s window. A failed or slow `/api/health` fetch counts as warming (the gateway function can be cold too). Pre-warm: `/api/warm` on home and play load, and on hover/focus of boss 3-5 tiles/tabs (one call per 8 s). There is no build-time flag.
Check by hand after the guard has been idle 15+ minutes (do not hit the API meanwhile, `/api/health` itself pings the guard): load the home page in a browser, confirm the nav pill says "guard waking up" and boss 3 shows BOSS WAKING UP…, never "offline".

Measured on production (2026-11-12, web/api b948fbe, guard v2; guard and gateway idle 17 minutes, real mobile browser at 390 px, session started and bosses 1-2 skipped):
| t after page load | event |
|---|---|
| 0.7 s | home page loaded, nav pill "checking" |
| 9.4 s | first `/api/health` answers after 8.6 s (gateway cold start plus the 2 s probe): `guard_state=warming` |
| 13.6 s | pill "guard waking up", no offline banner |
| 16.6 s | second health (5.2 s): still warming |
| 18.0 s | boss 3 selected: BOSS WAKING UP… screen visible, tab labelled "waking up", composer disabled, no "offline" anywhere |
| 22.4 s | third health (3.8 s): `guard_state=ok` |
| 22.6 s | pill "guard online", waking screen gone, composer enabled by itself: boss 3 playable about 22 s after the page loaded |
| 30.0 s | first boss 3 message answered, verdict PASSED (7.4 s round trip, mostly the vault) |
The first run of this check (before the client fix, ADR-0098 addendum) showed "guard offline" at 9 s because the browser's 8 s fetch limit expired while the gateway itself was cold; the table above is the run after the fix. One cold run, not a distribution.

## Vercel build rules: builds run only when the project's own code changed (ADR-0099)
Both projects use an Ignored Build Step (project setting `commandForIgnoringBuildStep`, exit 1 = build, exit 0 = skip). Non-main branches never build. On `main` a merge builds only when `git diff HEAD^ HEAD` touches the project's paths:
- `breachlab-web` (root directory `web`): anything under `web/` (`git diff --quiet HEAD^ HEAD -- .`).
- `breachlab-api` (repo root): `index.py requirements.txt src configs pyproject.toml uv.lock .vercelignore`. `configs/` is included because `configs/levels.yaml` is baked into the API.
- Docs-only, scripts-only, tests-only and report merges never build either project. The guard project is deployed by `scripts/deploy_vercel_guard.py`, not by git.
The commands are `[ "$VERCEL_GIT_COMMIT_REF" != "main" ] && exit 0; git diff --quiet HEAD^ HEAD -- <paths> 2>/dev/null; [ $? -eq 0 ] && exit 0; exit 1`: skip only when git positively reports no change in the paths; if `git diff` fails for any reason the build RUNS (fail-to-build). History: the first version skipped on error, and because `.vercelignore` removed `.git` from the checkout before the command ran ("Not a git repository"), it skipped every merge including a code merge (8fa7e9f). `.vercelignore` therefore no longer lists `.git` (only `.git/hooks` and `.git/logs`). Always check the build log of the first merge after a change to this rule (`GET /v3/deployments/<id>/events`) and after each code merge that a build started (`GET /v6/deployments?app=<name>`). The daily limit is 90 deployments over all projects (ADR-0101).
