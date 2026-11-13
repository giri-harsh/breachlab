# BreachLab — working notes (keep <=150 lines)

Source of truth: `PRD.md` (v1.0). Section 3 hard constraints override everything.

## Hard constraints (PRD s3)
- Cloud only: no model weights/training/inference in this container; those run on Kaggle/HF (transit-only relay of bytes is allowed, ADR-0040).
- No AI/co-author attribution anywhere (commits, PRs, files). Conventional Commits. Hook enforces it.
- Secrets only in env / Modal / GH secrets. Never commit, log, trace. detect-secrets pre-commit.
- Python 3.11+, mypy strict on `src/`, pydantic at boundaries, >=80% coverage (Modal entrypoints omitted).
- Budget caps in `configs/budget.yaml` ($25/job, $150/month). Log projected vs actual cost to MLflow.
- Ask Harsh only per PRD s3.8 (missing credentials, over-cap spend, DNS, irreversible public actions, license ambiguity).

## Architecture (PRD s6)
frontend (Next.js/Vercel) -> gateway (FastAPI on Vercel) -> input guard (Qwen3-0.6B LoRA, int4 ONNX, CPU, Vercel function, ADR-0082)
-> vault agent (Groq) -> output filter -> leak detector. Neon Postgres, DVC+MLflow on DagsHub, HF Hub.

## Workflow
- Branches `phase-N/<task>`; PRs; push after every task. Update this file + `docs/progress.md` each task.
- Long jobs run detached on Modal; write job ID to progress.md before polling.
- Decisions go in `docs/decisions.md`.

## Commands
- `uv sync --all-groups`; `uv run ruff check .`; `uv run mypy`; `uv run pytest`
- `uv run python scripts/check_env.py` (writes reports/env-check.json)

## Current state
- Phase 1: live checks done except CI cloud-smoke (needs GH Actions secrets: run scripts/set_gh_secrets.sh where gh works) and Modal GPU (payment method; Kaggle T4 is the GPU path, ADR-0020).
- Phase 2: data-v1 DONE (ADR-0030; DVC data/processed.dvc; HF Pseudo-Harsh/breachlab-data; branch `data-v1` stands in for the blocked git tag).
- Phase 3: 0.6B seed-0 SFT+DPO done, preliminary reports/eval-v1.md. Running Kaggle kernels: giriharsh/breachlab-sft-17b-s0, giriharsh/breachlab-sft-06b-s1 (see docs/progress.md). Next: fetch with `kaggle kernels output --file-pattern '.*\.(log|json|npy)$'` (NEVER download adapters), run stage evalgold (--from-kernel), re-run scripts/make_report.py, seed 2, ONNX int8 export + CPU latency, run scripts/run_api_baselines.py to finish Prompt-Guard and Gemini-judge rows.
- $0 budget; Modal is permanently out (ADR-0037). Guard on a Vercel function (ADR-0082; HF Spaces need PRO, ADR-0079). Sandbox egress is HTTPS-only.
- No Discord/phone alerts (ADR-0025); Grafana OTLP HTTP basic auth (ADR-0026); Kaggle auth KAGGLE_API_TOKEN (ADR-0024).
## RESUME HERE (updated 2026-11-13: BreachLab v1.0.0 SHIPPED; only v2 backlog remains)
- Shipped: public repo https://github.com/giri-harsh/breachlab (v1.0.0 notes in docs/RELEASE_v1.0.0.md; the tag and GitHub release must be created by Harsh: the sandbox refuses release writes and tag pushes), live game https://breachlab-web.vercel.app, HF model (Apache-2.0) and dataset (CC-BY-SA-3.0) PUBLIC. Prod: web/api 97f551d, guard v2 (`champion-v2/`, MLflow breachlab-guard v3 = champion; rollback = GUARD_ALIAS=champion + guard redeploy). Neon at Alembic 0002 (DVC backup backups/neon-20261112).
- Repos: this private repo `giri-harsh/Breach-lab` is the ARCHIVE (history). Both Vercel projects build from the PUBLIC repo: to deploy, sync a commit to `git remote public` (commit-tree on top of public main from this repo's tree, `scripts/prepublic_scan.py --tree DIR` must say SAFE, push). Ignored Build Step is path-scoped and fails to build on error (ADR-0099); deploy limit 90/day (ADR-0101).
- CI: GitHub account billing lock fails every Actions job (ADR-0104). All jobs are gated on `vars.ACTIONS_BILLING_OK == 'true'` (unset => skipped, no red runs); secret-needing workflows also skip without secrets. Merge gate stays the LOCAL gate (ADR-0092): ruff, format, mypy, pytest >=80% cov, web lint/typecheck/build, Playwright, prepublic_scan, no AI trailers. Switch-on steps are in ADR-0104.
- V2 BACKLOG: (1) review of the 393 disagreeing gold rows; (2) Hindi opener FPR 15.0% at its gate (more Hindi greetings, v3 data); (3) latency: 300 ms bar missed (p95 2,063 ms warm, cold ~13 s): faster host or a GPU path; (4) custom subdomain (DNS); (5) fresh out-of-distribution test set from real players; (6) Grafana snapshot + alerts; (7) boss 2/3 difficulty depends on the Groq vault model (200k tokens/day cap): re-measure when quiet.
- Rules: no AI trailers; Conventional Commits; migrations BEFORE the code that needs them; $0; never touch v1 guard weights; secrets never printed; Groq quota is shared by vault, data generation and leak suites (do not run suites in bulk against production).
- Still Harsh: fix GitHub billing (then follow ADR-0104), create tag v1.0.0 and publish the release (notes: docs/RELEASE_v1.0.0.md).
