# Lane C report: GATEWAY AND FLYWHEEL

LANE: C (gateway, flywheel, drift untouched)

TASKS_DONE
1. Hot-swap: `src/breachlab/gateway/serving.py` (ServingState, poll-on-request TTL 30 s, REST + MlflowClient sources, optional thread, `serving_version()`); wired in `service.py` (one snapshot per request, stamps `attempts.guard_version`, telemetry, leaderboard/hall) and `app.py` (`/api/health` also returns `serving_version`). Tests `tests/gateway/test_serving.py` (swap, rollback, MLflow down keeps old champion, swap mid-request, single refresher).
2. Fail closed: L3/L4/L5 guard unreachable, slow or malformed -> 503 "guard waking up", rollback, `guard_unavailable_total{level,alias,kind}` (METRICS + OtelMetrics + `Telemetry.guard_unavailable`). Fixed a fail-open bug: the L5 output filter swallowed guard errors. `HttpGuard` parses `p_unsafe` + `confidence` (tolerates old servers), `GUARD_TOKEN` bearer, injectable client. Tests `tests/gateway/test_failclosed.py`.
3. Contract tests: `tests/gateway/test_guard_contract.py` + `fake_guard_server.py`; runs against the fake always and against `GUARD_URL` (+ optional `GUARD_TOKEN`) when set (9 live variants skip here).
4. Flywheel: `flywheel/judge.py` (Gemini-style batch judge via LLMClient, PRD 10 rules, review queue, Cohen kappa), `live.py` (trust rules, PII scrub, selfplay flag, canary/gold excluded), `build_dataset.py` (dedup vs existing splits, gold_test dup drop, check_leakage, train/val by session, CLI), `fresh_test.py` (newest whole sessions, disjoint by parent group). Tests `tests/flywheel/`.
5. Self-play: `flywheel/selfplay.py` (<=500 sessions, injected HTTP and LLM clients, pacing, 429/503 backoff, `selfplay-` nicknames; command in module docstring). Mock tests incl. end-to-end against `create_app` then judge then dataset.
6. Public routes unchanged (health gains one optional field).

TESTS: ruff clean, `mypy` strict clean (78 files), pytest gateway+flywheel (ex test_monitor) 120 passed 9 skipped (live contract), coverage of src/breachlab/gateway + flywheel 99%. Full suite run result in the final message.

BRANCH: lane-c/gateway-flywheel (commits 0116d09, 60d8638, plus the docs commit on top)

REQUESTS (orchestrator-owned files)
- docs/CONTRACTS.md s2: append "`GET /api/health` also returns `serving_version` (e.g. `v2`) when the registry source is configured. Metric `guard_unavailable_total` counts fail-closed guard errors."
- Vercel env for breachlab-api (via scripts/set_vercel_env.sh): `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` (hot-swap off without them), `GUARD_TOKEN` only if the Space is private, optional `CHAMPION_TTL_S` (default 30), `GUARD_TIMEOUT_S` (default now 3.0).
- Dashboards/alerts (Lane D/E or orchestrator): add `guard_unavailable_total` panel and a "rate > 0 for 5 m" alert.
- Lane B: `/health.models.champion` should follow the same registry alias so it matches `serving_version`.
- configs/*.yaml: none. pyproject/uv.lock: none (flywheel uses the existing `data` group: datasketch, pandas, pandera).

ADR_DRAFTS: docs/adr-drafts/lane-c-champion-hotswap.md, lane-c-guard-fail-closed.md, lane-c-flywheel-labeling.md, lane-c-selfplay.md

BLOCKED: none for this lane. (Live runs need Gemini/Groq keys and a live game: Lane F.)

RUNNING_JOBS: none

NEXT: Lane F runs `python -m breachlab.flywheel.selfplay --base-url <gateway> --sessions N`, then the label pass `python -m breachlab.flywheel.judge` (GEMINI_API_KEY, NEON_DATABASE_URL; GeminiPool judge, prints stats and judge-human kappa) and `python -m breachlab.flywheel.build_dataset --existing data/processed --out data/live-v2 --fresh-n 300`. Neither CLI was run here (no quota).
