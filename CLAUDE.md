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
## RESUME HERE (orchestrator, updated 2026-11-12 ~20:00 UTC; H1-H4 batch done; main = 97f551d + final docs, LIVE)
- Public repo: https://github.com/giri-harsh/breachlab (single clean snapshot commit b70b25d + doc follow-ups, scan SAFE, ADR-0100). BOTH Vercel projects are now linked to the PUBLIC repo: merging to this private repo builds nothing; to deploy, push to the public repo (`git remote public`; build a commit-tree on top of the public main from this repo's tree, run `scripts/prepublic_scan.py --tree`, push). This private repo `giri-harsh/Breach-lab` is the ARCHIVE (full history). Work continues here; to publish updates, rebuild a snapshot (orphan commit from main, `scripts/prepublic_scan.py` must say SAFE) or sync the public repo. Actions on the public repo is blocked by the GitHub ACCOUNT billing lock (needs Harsh); the local gate (ADR-0092) stays until a run goes green.
- Prod: web/api 97f551d code, guard serves `champion-v2/` (MLflow breachlab-guard v3 = champion; rollback = GUARD_ALIAS=champion + guard redeploy). Neon at Alembic 0002 (DVC backup backups/neon-20261112). Deploy rule ADR-0101: <= 90 Vercel deployments/day over all projects incl. canceled (count first), batch code merges. Vercel Ignored Build Step is path-scoped and FAILS TO BUILD on error (ADR-0099 addendum); after each code merge confirm a build started.
- Difficulty (prod, ADR-0102, docs/lanes/G-difficulty-prod.json): boss 1 fresh 79.1%, boss 2 fresh 57.3% (template 51.4% replay), boss 3 pooled 21.6%, bosses 4/5 0/326. 0% on templated attacks is NOT unbreakable. Groq gpt-oss-20b has a 200k tokens/day cap shared with players: do not run leak suites in bulk against production.
- Open: Hindi opener FPR 15.0% at its gate (v3 data target); boss 2/3 difficulty depends on which vault model answers.
- Rules: no AI trailers; Conventional Commits; apply DB migrations BEFORE merging code that needs them (runbook); $0; never touch v1 guard weights; secrets never printed; Groq quota is shared by vault, data generation and leak suites.
- Still Harsh: GitHub account billing lock (blocks Actions everywhere). HF repos are still private; making them public is his call.
