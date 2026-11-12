# ADR draft (lane P): full request latency, cost per 1k, rollback timing and reproducibility tags (2026-11-12)

## Context
The PRD gap check (`docs/prd-gap.md`) listed four $0 items as MISSING/PARTIAL for lane P: full request p95 (PRD 1.2,
never measured), cost per 1k requests (PRD 9.1, `cost_usd` never populated), a stored rollback timing (PRD 10.8; the
runbook said "about 1.2 s" without a record) and reproducibility tags (PRD 12). No deploys, no Kaggle jobs, no push.

## Measurements
1. **Full request, warm, level 4, production** (`scripts/p_full_request_latency.py`, `docs/lanes/P-full-request.json`):
   n=100 (50 benign / 50 attacks), concurrency 1, 5 canary sessions, 4 s pacing, 0 errors.
   Client p50 **5053 ms**, p95 **6447 ms**, max 7912 ms. Server-side p50 3186 ms, p95 3887 ms.
   Guard-only (blocked) server p50 3090 ms vs guard + vault 3289 ms. **The 2 s target is not met**; the guard call path
   dominates (consistent with ADR-0082's 3468 ms guard model p95), not the Groq vault.
2. **Cost per 1k requests** (`docs/lanes/P-cost.json`): actual bill $0 (free tiers). Vault tokens estimated from lengths
   (API exposes no usage): ~23.9k input + ~73.6k output per 1k requests (45% reach the vault; lower bound, hidden
   reasoning excluded) -> ~$0.024 at Groq gpt-oss-20b list price $0.075/$0.30 per 1M (console.groq.com/docs/models,
   fetched 2026-11-12). Guard CPU-seconds upper bound 3.23 s/request (~3229 s per 1k), not priced.
3. **Rollback** (`scripts/p_rollback_timing.py`, `docs/lanes/P-rollback-timing.json`), separate model
   `breachlab-guard-rollback-test` on DagsHub, 3 reps: promote 0.64-3.16 s, rollback 1.95-5.70 s (median 4.88 s), max
   5.85 s with read-back. Plus the 30 s alias TTL -> worst case ~36 s, **< 1 min met**. The runbook's "about 1.2 s" is
   corrected to "2-6 s on DagsHub".

## Decision / changes
- New `breachlab.train.repro`: git SHA (env -> staged GIT_SHA -> git rev-parse -> "unknown"), sha256 of the resolved
  TrainConfig without machine paths, `data-vN`, data file hash, seed. Stored in `results.json["repro"]`, staged into the
  Kaggle code dataset, set as MLflow tags by the relay. Unit tests without network.
- Two measurement scripts (`scripts/p_*.py`) with unit tests for their pure parts.

## Consequences / follow-ups
- PRD 1.2 full-request p95 stays a documented miss alongside the 300 ms guard miss; the fix is the same (a faster guard
  host, money) plus fewer gateway DB round trips. Worth exposing guard latency_ms and vault tokens in the trace to
  replace the estimates.
- Runbook line on rollback ("about 1.2 s") should read "2-6 s alias move on DagsHub, takes effect within the 30 s TTL"
  (runbook not owned by lane P).
- `scripts/register_runs.py` should call `log_repro_tags`; existing registered versions predate the tags.
- Benign false positives in the run (5/50) were all the known Hinglish probe (ADR-0084/0090, v2 data).
