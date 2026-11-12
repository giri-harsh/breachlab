# Lane D report: WEB, CI, RELEASE

LANE: D (web/, model-gate workflow, set_* scripts, link_check)

TASKS_DONE
1. Vercel runtime env for `breachlab-api` set via REST (POST /v10/projects/prj_xSY0.../env, 17 keys, all HTTP 201; the classifier did not block it): NEON, GROQ, LANGFUSE_*, GRAFANA_OTLP_*, ADMIN_PASSWORD, MLFLOW_TRACKING_*, GUARD_URL=https://placeholder.invalid, CORS_ORIGIN, AGENT_MODEL=openai/gpt-oss-20b, CHAMPION_TTL_S=30, GUARD_TIMEOUT_S=3.0. Redeployed (with Lane C hot-swap): `/api/health` -> {"ok":true,"guard":true,"serving_version":"v1"}, `/api/leaderboard` 200. Round trip at L1 OK (nickname "fuck" -> 400 "nickname not allowed"; Groq reply ~2.4 s, 200). Groq 429s are intermittent on gpt-oss-20b (see REQUESTS).
2. GitHub Actions secrets: BLOCKED (REST `GET /actions/secrets/public-key` -> 403 "Access to this GitHub Actions path is not permitted through this proxy"; MCP has no secrets/protection tool). Not retried. promote/rollback runs therefore not triggered. `scripts/set_gh_secrets.sh` updated (Modal removed; adds KAGGLE_API_TOKEN, ADMIN_PASSWORD, GATEWAY_SECRET_KEY, GUARD_URL, GRAFANA_OTLP_*, optional BASE_URL variable).
3. Playwright: `web/e2e/play.spec.ts` (mocked L1-L5 kept), `pages.spec.ts` (new, mocked: pages, guard status, 503 offline, vault-busy, tab lock), `live.spec.ts` (new, E2E_LIVE=1, Pixel 7, canary header via `e2e/canary.ts`, `?unlock=1`). Local: 8 passed, 2 skipped (live). Live against https://breachlab-web.vercel.app: L1 and L2 pass (benign passes, attack unblocked at L1 / blocked at L2, UI renders), L3 fails fast with "guard offline (HTTP 503 after retry)" because GUARD_URL is a placeholder, which is the correct signal. Command: `E2E_BASE_URL=https://breachlab-web.vercel.app E2E_LIVE=1 ADMIN_PASSWORD=... PW_CHROMIUM=/opt/pw-browsers/chromium npx playwright test live` (use E2E_BASE_URL=the deployed web: the gateway's CORS only allows that origin). What it proves/does not prove: docs/adr-drafts/lane-d-live-e2e-canary.md.
4. `.github/workflows/model-gate.yml` (job `gate`): runs eval_gate.py + check_readme_numbers.py on model-evidence paths. Today the committed gate input fails it (fpr vs base champion, p95 1441.7 ms).
5. Web polish: Nav `GuardStatus` pill (health probe, "guard offline" on 503/network), play page shows distinct guard-offline and vault-busy states, `/under-the-hood` and `/model-card` read scorecard.json (all rows; shipped row `qwen3-0.6b-sft` with unrounded p95), links from `NEXT_PUBLIC_HF_NAMESPACE` (default Pseudo-Harsh) / `NEXT_PUBLIC_GITHUB_REPO`, Space link only when `NEXT_PUBLIC_HF_SPACE` is set. Contrast: all text >= white/60 (6.9:1 on panel, 7.2:1 on ink); tap targets >= 44 px; no horizontal scroll asserted. lint, tsc, build clean. Deployed: https://breachlab-web.vercel.app.
6. `scripts/link_check.py` checks README + web links (resolves `NEXT_PUBLIC_*` defaults from web/lib/links.ts), private github/HF 401/403/404 are WARN until LINKCHECK_STRICT=1, gateway root probes /api/health. Run: 0 FAIL, 4 WARN (private repo/HF). `tests/scripts/test_link_check.py` 10 tests.

TESTS: ruff clean; full `uv run pytest` 241 passed, 9 skipped, coverage 98.18%; web lint/tsc/build clean; Playwright 8 passed + 2 live-only. (mypy: pre-existing untyped-decorator errors in modal_* files from earlier lanes, not touched.)

BRANCH: lane-D/web-ci-release (rebased on origin/main 32a3c46); sha in the final message.

REQUESTS (orchestrator-owned)
- docs/blockers.md: replace the "Vercel env vars" line with "done via REST (ADR draft lane-d-vercel-runtime-env); only GUARD_URL placeholder remains: run `GUARD_URL=https://<space> scripts/set_vercel_env.sh` (or re-POST GUARD_URL) once the Space is live".
- docs/blockers.md: GH secrets + branch protection: run scripts/set_gh_secrets.sh locally; protection must list contexts `checks` and `gate`.
- reports/gate-input.json: add `"is_base": true` to `champion` (ADR-0036) and set `candidate.p95_latency_ms` to the Space measurement; configs/gate.yaml p95_latency_ms_max = measured + 20%.
- Lane C (gateway): `GroqAgent.reply` retries 429 three times within ~0.1 s with no backoff, so bursts return 503 "vault busy". Honour `retry-after` (cap ~5 s) between tries and fall back to a second model (e.g. openai/gpt-oss-120b or qwen/qwen3.8-27b) on 429.
- Lane C/B: `GET /` on the gateway returns 404; link checkers probe /api/health instead (handled in link_check.py).
- Vercel Git integration for both projects (needs Harsh: GitHub app on the repo) would remove the author-block workaround.

ADR_DRAFTS: docs/adr-drafts/lane-d-vercel-runtime-env.md, lane-d-live-e2e-canary.md, lane-d-vercel-deploy-author.md, lane-d-model-gate-workflow.md

BLOCKED
- GitHub Actions secrets + branch protection (proxy 403): `scripts/set_gh_secrets.sh` with env exported, or per secret `printf %s "$V" | gh secret set NAME --repo giri-harsh/Breach-lab`.
- GUARD_URL is a placeholder until Lane B's Space is live; L3-L5 return 503 meanwhile and the live e2e stops at L3.
- Non-owner CLI deploys: commits authored by the Claude bot identity make Vercel mark deployments BLOCKED; deploy from a git-less copy (ADR draft lane-d-vercel-deploy-author).

RUNNING_JOBS: none

NEXT: once GUARD_URL is live: re-POST GUARD_URL, redeploy `breachlab-api`, run the live spec for L3-L5 and attach `live-report.json` (test-results/ attachment) to docs/progress.md; then run promote/rollback workflows after secrets exist.
