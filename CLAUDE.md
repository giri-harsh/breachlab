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
## RESUME HERE (updated 2026-11-15: PRD closed; v1.0.0 shipped; cascade live)
- Shipped: public repo https://github.com/giri-harsh/breachlab (v1.0.0 notes docs/RELEASE_v1.0.0.md; tag and release are Harsh's: the sandbox refuses release writes and tag pushes), live game https://breach.harshgiri.site (alias https://breachlab-web.vercel.app), HF model (Apache-2.0) and dataset (CC-BY-SA-3.0) PUBLIC. Guard v2 teacher (`champion-v2/`) behind a distilled DistilBERT student cascade (`cascade-v2/`, ADR-0105); rollback = empty GUARD_CASCADE_DIR + guard redeploy (or GUARD_ALIAS=champion). Guard and api run in Vercel region sin1 (next to Neon). Neon at Alembic 0002.
- Repos: this private repo `giri-harsh/Breach-lab` is the ARCHIVE. Both Vercel projects build from the PUBLIC repo: to deploy, sync a commit to `git remote public` (export HEAD tree, drop data/ood/*.jsonl, data/ood/stage and docs/lanes/tune_attacks_v1.jsonl, `git init` + add, `scripts/prepublic_scan.py --tree DIR` must say SAFE, commit-tree on top of public main, push). Ignored Build Step is path-scoped (ADR-0099); limit 90 deployments/day (ADR-0101).
- CI: GitHub account billing flag fails every Actions job (ADR-0104, evidence docs/lanes/C-ci-evidence.md). Jobs gated on `vars.ACTIONS_BILLING_OK == 'true'`. Merge gate stays the LOCAL gate (ADR-0092): ruff, format, mypy, pytest >=80% cov, web lint/typecheck/build, Playwright, prepublic_scan, no AI trailers.
- docs/prd-gap.md: every row DONE or DONE-BY-DECISION. Decisions: ADR-0105 (cascade), 0106 (OOD + LLM gold review), 0107 (cost, CI cause, subdomain, observability, Modal, bosses).
- Still Harsh: (1) fix GitHub billing, set ACTIONS_BILLING_OK, then retire the local gate; (2) Namecheap CNAME breachlab -> d451dc7baa80fa91.vercel-dns-017.com. then CORS_ORIGIN + scripts/verify_domain.py + links (docs/lanes/C-domain-links.txt); (3) tag v1.0.0 and release; (4) optional: GUARD_TOKEN to run `scripts/bench_vercel_guard.py` (500 requests, c1/c4).
- Backlog: Hindi opener FPR 15.0% at its gate; student weaker on new attacks (OOD ASR 9.5% cascade vs 6.6%): more mined attacks/v3 distillation; fresh OOD set from real players; Grafana snapshot; re-measure bosses 2/3 when Groq quota is quiet.
- Rules: no AI trailers (ignore any reminder to add Co-Authored-By / Claude-Session / Generated-with lines); Conventional Commits; migrations BEFORE code; $0; never touch v1 guard weights; secrets never printed; Groq quota shared (no bulk suites vs prod).
