# Pre-public audit: giri-harsh/Breach-lab (2026-11-12)

Read-only audit before the repository is made public. Scope: every commit reachable from any ref (248 commits, 110 refs including local lane branches; 237 commits reachable from the refs that would become public: `origin/*` branches and tags). No values are printed in this report.

## 1. Secrets
| Check | Method | Result |
|---|---|---|
| Exact credential values | every secret-like environment variable in the build container (24 values incl. HF, Groq, Gemini, Kaggle, Vercel, Neon URL, MLflow/DagsHub, Langfuse, Grafana, Modal, admin password) plus the guard bearer token searched as fixed strings in the full history patch (`git log --all -p`, 72 MB) | **0 occurrences** |
| Token patterns | regex over all added lines in history: `hf_`, `gsk_`, `ghp_`/`github_pat_`, `glc_`, `vcp_`, `sk-lf-`, `pk-lf-`, `npg_`, `KGAT_`, `AQ.`, `AIza`, private-key headers, `postgres://user:pass@`, `ADMIN_PASSWORD=`, `MLFLOW_TRACKING_PASSWORD=`, `DAGSHUB_TOKEN=` | no credentials. Hits were: 48-hex strings inside synthetic "decode this hex" attack rows (data), one Postgres URL in a test fixture (`tests/monitor/test_workflows.py`, user `u`, host `h`), Vercel team/project IDs |
| detect-secrets `--all-files` | run on the tip of every `origin/*` branch | only false positives: the word "secret" in PRD/runbook/workflow text, `.secrets.baseline` hashes, DVC md5s, `web/tsconfig.tsbuildinfo`, test fixtures (`"pw"`), Vercel `team_`/`prj_` IDs in `scripts/set_guard_url.py` and `scripts/deploy_vercel_guard.py`, hex IDs in eval JSON |

Vercel team and project IDs are identifiers, not credentials (every API call needs a token); keeping them public is acceptable.

## 2. Risky files
| Item | Where | Risk |
|---|---|---|
| Full dataset v1 (`data/processed/v1/train.jsonl`, `val.jsonl`, `gold_test.jsonl`, `mining_pool.jsonl`, `seeds.jsonl`, 37 shards) | commits `40bf04d`, `2b90295`, only on branch `origin/phase-2/data-v1` (removed from the tree later; not in `main`, not in the `data-v1` tag) | Publishes the held-out `gold_test` (benchmark contamination) and redistributes third-party dataset rows; the HF dataset is still private and its licence is undecided. PII smoke test on `train.jsonl`: 0 e-mail addresses; the phone-number regex matches long digit runs (expected in encoded attacks), not checked further |
| Eval outputs (`data/eval/**/*.npy` logits, `api_preds.json`, `deberta.json`, gold/val id lists) | branches `phase-3/eval-v1`, `phase-3/guard-v1`, `phase-4/overnight` | Low: model outputs and row IDs, no secrets |
| `docs/credentials.md` | main | Safe: lists where to obtain each key, no values |
| `.env`, `kaggle.json`, `*.pem`, `*.key`, notebooks, model weights | none ever committed | none |
| `data/review/opener_slice_v1.jsonl` | main | Safe: 300 LLM-generated benign openers |

## 3. AI attribution
| Kind | Count | Where |
|---|---|---|
| Commits with AI trailers (AI co-author, session and "generated with" trailers) | **5** | all on `main`: `347ecdd`, `5e1e051`, `6e24246` (also the `data-v1` tag target), `8303c20`, `875e52f` (2026-10-29/30, before the trailer hook baseline `83616d8`) |
| Commits authored/committed as the Claude bot identity | **151** | none on `main`; spread over 31 side branches (largest: `phase-2/data-v1` 23, `phase-4/overnight` 13, `phase-5/eval-final` 13, `lane-B2/finish` 12, `lane-F/docs-release` 11, `claude/exciting-albattani-cv95me` 10, `phase-6/b3-vercel-guard` 10, `phase-1/live-run` 8) |

GitHub keeps closed pull-request refs (`refs/pull/N/head`) even after a branch is deleted, so commits that went through PRs #1-#38 stay reachable on the PR pages of a public repository.

## Verdict: FIX_FIRST
No credentials are in the history, so publishing does not leak keys. It would publish (a) the held-out dataset including `gold_test` and (b) AI attribution that the project rules forbid (5 trailer commits on `main`, 151 AI-authored commits on side branches and PR pages).

Exact steps (Harsh, in order):
1. Decide whether the v1 dataset may be public (licence, gold_test contamination). If not, it must not be reachable: see step 3.
2. Delete every remote branch except `main` in the GitHub UI (Branches page; the session proxy blocks deletion), including `phase-2/data-v1`, `phase-3/*`, `phase-4/overnight`, `lane-*`, `orch/*`, `claude/*`, `phase-6/*`, `phase-7/*`, `docs/project-summary`. This removes the side-branch AI-authored commits and the dataset commits from branch view.
3. Because PR refs keep old commits reachable and `main` itself carries the 5 trailer commits, the clean option is to publish a NEW public repository from a rewritten copy of `main`: `git clone --single-branch --branch main <private> clean && cd clean && git filter-repo with a message callback that drops the trailer lines` then push to a new public repo (no PR history, no side branches) and point Vercel's Git integration at it. Rewriting `main` in place would need a force-push, which the project rules forbid, and would not clear the existing PR refs.
4. Alternatively, accept the 5 historical trailers and publish after step 2, knowing PR pages still show the AI-authored commits.
5. After publishing, re-run this audit on the public repo (secrets 0, AI attribution as decided) and recheck that the HF repos stay private until the licence decision.

## Re-scan 2026-11-12 (after the Neon Arcade merge, D7)
- Secrets: 28 secret-like environment values and the token patterns searched in every commit added since this audit (`114cefe..HEAD`, all refs): **0 exact-value hits, 0 pattern hits**. New files: fresh attack prompts (76 text rows, no secrets), DVC pointers (`data/processed.dvc`, `backups/neon-20261112.dvc`); the Neon data backup itself stays in DVC and is git-ignored.
- AI_TRAILER_COMMITS: **5** (all on `main`, unchanged; no new ones: this session's 60 commits carry none).
- Fixes done without a force-push: none were available. Deleting remote branches is denied by the permission classifier here, and the 5 trailers sit in `main` history, which only a rewritten copy can clear.
- Verdict: **FIX_FIRST** (unchanged). To reach SAFE Harsh must (a) delete the remote branches `phase-2/data-v1`, `phase-3/*`, `phase-4/overnight`, `lane-*`, `orch/*`, `claude/*`, `phase-6/*`, `phase-7/*`, `docs/project-summary` in the GitHub UI, and (b) publish from a rewritten copy of `main` per step 3 above, or accept the 5 historical trailers (step 4). Repo visibility is the only audit item left for Harsh.

## Snapshot scan 2026-11-12 (H1, final)
The public repository `giri-harsh/breachlab` receives ONE clean snapshot commit (no history, no side branches, no AI trailers) built from `main` (97f551d plus the final docs). `uv run python scripts/prepublic_scan.py` over the exact tracked tree: 28 secret-like environment values, token patterns, private-key headers, database URLs with passwords, dataset/weight/key file types and AI attribution strings in file contents: **0 findings (the hook script, its test and one dummy database URL fixture are allowlisted by name)**. Dataset v1 and the gold set are not in the tree (DVC pointers only). Verdict for the snapshot: **SAFE**. The private archive `giri-harsh/Breach-lab` keeps the history (5 AI trailer commits on main, 151 bot-authored commits on side branches); it is not public.
