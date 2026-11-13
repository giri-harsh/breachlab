# Local development

Everything runs on a laptop without paid services. The guard model itself is never run locally: levels 3-5 call a guard host over HTTP (`GUARD_URL`), or you play levels 1 and 2 without it.

## Prerequisites
Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 22, and (optional) a free Groq API key for the vault agent.

## 1. Checks (what CI would run)
```bash
uv sync --all-groups
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
uv run python scripts/check_readme_numbers.py      # README numbers vs web/public/scorecard.json
uv run python scripts/prepublic_scan.py            # secrets / risky files / AI attribution in tracked files
cd web && npm ci && npm run lint && npm run typecheck && npm run build
```
The Python suite needs no network. The web suite `npx playwright test` runs against mocked APIs (`E2E_BASE_URL` points it at a deployment; `E2E_LIVE=1 E2E_REAL=1` add the specs that need a real gateway).

## 2. Run the gateway
```bash
export NEON_DATABASE_URL=sqlite:///./local.db     # any SQLAlchemy URL; Postgres in production
export ADMIN_PASSWORD=dev-only-password
export GROQ_API_KEY=...                           # free tier; without it levels 1-2 cannot answer
# optional, for levels 3-5: a running guard host (see docs/deploy-guard.md)
# export GUARD_URL=https://<guard host>  GUARD_TOKEN=...
uv run uvicorn index:app --port 8000
curl -s localhost:8000/api/health                 # {"ok":true,"guard":...,"guard_state":"ok|warming|down"}
```
Level 2's keyword wall and the levels config live in `configs/levels.yaml`; the vault prompts, hints, points and per-level guard thresholds are all there.

## 3. Run the web app
```bash
cd web
echo 'NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000' > .env.local
npm run dev        # http://localhost:3000
```

## 4. Useful scripts
| script | what it does |
|---|---|
| `scripts/run_leak_suite.py <level>` | runs the 216 template attacks (or `--attacks file.jsonl`) against an in-process gateway; resumable |
| `scripts/run_prod_attacks.py` | the same attacks against a deployed gateway with canary sessions (spends vault quota) |
| `scripts/v2_gates.py` | guard v2 gates from Kaggle logits |
| `scripts/make_hf_cards.py --dry-run` | generates the Hugging Face cards into `reports/hf-cards/` |
| `scripts/deploy_vercel_guard.py` | deploys the guard function (needs Vercel and Hub tokens) |

Training, export and quantisation run as Kaggle kernels (`src/breachlab/train/kaggle_job.py`, `export_job.py`, `serve/quant_job.py`); weights never enter the dev container. Operations are in `docs/runbook.md`, decisions in `docs/decisions.md`.
