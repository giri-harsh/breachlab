LANE P (PRD gaps: full request p95, cost per 1k, rollback timing, reproducibility tags) | 2026-11-12
Branch `phase-8/p` (local, from `phase-8/arcade`). ADR draft: `docs/adr-drafts/P-prd-gaps.md`.

## 1. Full request p95 (PRD 1.2: <= 2 s warm): NOT MET
`uv run python scripts/p_full_request_latency.py` (ADMIN_PASSWORD in env; canary HMAC header, key never printed).
Production https://breachlab-api.vercel.app, level 4, 2026-11-12 05:30 UTC, guard warm (`/api/warm` = warm, 2
unrecorded warm-up messages), 100 requests, concurrency 1, 5 canary sessions, starts 4 s apart, mix 50 benign / 50
attacks, interleaved. 0 errors, guard_version v1 on all 100.
- Client end to end (sandbox -> Vercel): p50 5053 ms, p95 6447 ms, max 7912 ms.
- Server-side `latency_ms` (inside the gateway): p50 3186 ms, p95 3887 ms, max 6071 ms.
- Guard-blocked (no vault call) server p50 3090 ms vs passed (guard + Groq) 3289 ms: the vault adds about 0.2 s;
  the guard call path (API function -> guard function, plus DB writes/rate-limit queries) dominates. Client minus
  server overhead median 1854 ms (sandbox egress proxy + Vercel routing; a browser near the region would see less).
- Side observations: 0 leaks; all 50 attacks blocked; 5/50 benign blocked, all the known Hinglish
  "Mujhe cricket ke rules samjhao." false positive (ADR-0084/0090).
Result: `docs/lanes/P-full-request.json` (per-request records, no reply texts).

## 2. Cost per 1k requests (PRD 9.1)
Same run, `docs/lanes/P-cost.json`. The public API exposes neither Groq usage nor the guard's own latency_ms, so:
- Vault tokens ESTIMATED (chars / 4, L4 system prompt included; gpt-oss hidden reasoning not visible, lower bound):
  45% of requests reach the vault; per 1k requests about 23.9k input + 73.6k output tokens.
- Guard CPU-seconds UPPER BOUND from server latency of blocked requests: 3.23 s/request, about 3229 s per 1k.
- Actual bill: $0 (Groq free tier, Vercel Hobby, Neon free).
- USD list price: Groq gpt-oss-20b $0.075 / $0.30 per 1M in/out tokens, from https://console.groq.com/docs/models
  fetched 2026-11-12 (groq.com/pricing renders client-side and gave no prices) -> about $0.024 per 1k requests for the
  vault (lower bound). Guard CPU: not priced.

## 3. Rollback timing (PRD 10.8: < 1 min): MET for the alias move
`uv run python scripts/p_rollback_timing.py` on a separate registered model `breachlab-guard-rollback-test` (two dummy
versions pointing at a tiny JSON, no weights) on DagsHub MLflow; refuses `breachlab-guard`. 3 cycles of
`registry.promote(v2)` + `registry.rollback()`, each with an alias read-back, 2026-11-12 05:33 UTC:
- promote 0.64 / 2.99 / 3.16 s; rollback 1.95 / 4.88 / 5.70 s (median 4.88 s); rollback incl. read-back max 5.85 s.
- The runbook's "about 1.2 s" is CORRECTED: a DagsHub alias rollback takes 2-6 s (it is 2-4 REST calls at about
  0.6-1.5 s each). With the gateway alias TTL (30 s, `serving.DEFAULT_TTL_S`) the worst case is about 36 s < 60 s.
  Caveat as before: the Vercel guard serves `GUARD_ONNX_FILE`, so a graph change still needs a redeploy.
Result: `docs/lanes/P-rollback-timing.json`. The test model is left in the registry (harmless, makes it re-runnable).

## 4. Reproducibility (PRD 12)
Before: MLflow runs (`scripts/register_runs.py`, `train/modal_train.py`) logged kernel, base_model, `data: data-v1`
and seed as params; no git SHA, no config hash; `pipeline.run` results.json had tag/base_model/seed only.
Now (`src/breachlab/train/repro.py`, tests in `tests/train/test_repro.py`, no network):
- `git_sha`: BREACHLAB_GIT_SHA / GITHUB_SHA -> staged `GIT_SHA` file -> `git rev-parse HEAD` -> "unknown" (never raises).
- `config_sha256`: sha256 of the resolved TrainConfig as canonical JSON, excluding data_dir/output_dir (machine paths).
- `data_version` (`data-v1`), `data_sha256` over train/val/mining_pool when present, `seed`, `base_model`.
- `kaggle_job.stage_code` writes `GIT_SHA` into the code dataset (kernels have no .git); `pipeline.run` stores the
  tags under `results.json["repro"]`; `modal_train.relay` sets them with `mlflow.set_tags`; `log_repro_tags(cfg)` and
  `missing_tags(tags)` are the helpers for any other logger.
Open (not my files): `scripts/register_runs.py` should call `log_repro_tags` (or set tags from results.json["repro"]);
existing registered versions 1-2 of `breachlab-guard` predate the tags and were not modified.

Checks: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy` (clean), `uv run pytest` 537 passed,
coverage 98.2%.
