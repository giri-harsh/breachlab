# Frozen interfaces (orchestrator-owned; change only through an ADR)

## 1. Guard service (Vercel function today, ADR-0082) — `POST {GUARD_URL}/score`
Request: `{"text": str, "kind": "input"|"output", "alias": "champion"|"challenger"|"base"}`
Response 200: `{"label": "SAFE"|"UNSAFE", "category": "none"|"injection"|"jailbreak"|"exfiltration",
"p_unsafe": float, "confidence": float, "version": str, "latency_ms": float}`; `confidence` equals `p_unsafe`
(kept for the gateway's `HttpGuard`). Unknown alias falls back to `champion`. `GET /health` ->
`{"ok": true, "models": {"champion": ver, "challenger": ver|null, "base": ver|null}}`.
Python protocol: `breachlab.gateway.guards.Guard.score(text, kind, alias) -> GuardOutcome`.
Scoring maths: one forward pass, softmax over the 4 verbalizer logits (`configs/serve.yaml`) / T,
`p_unsafe = 1 - p_safe`, UNSAFE if `p_unsafe >= threshold`, category = argmax of the 3 unsafe classes.

## 2. Gateway routes (FastAPI, `breachlab.gateway.app`)
`GET /api/health` (also returns optional `serving_version` when a registry source is configured) | `POST /api/session {nickname}` (header `X-Canary: <ts>.<hmac>` marks a canary session) |
`GET /api/session/{id}` | `POST /api/message {session_id, level 1-5, message<=500 chars}` ->
`{blocked, block_category, reply, leaked, level_cleared, messages_left, guard_version, latency_ms}` |
`GET /api/leaderboard` -> `{leaderboard[], hall_of_breaches[]}` | `GET /api/admin/queue`, `POST /api/admin/label`
(`Authorization: Bearer ADMIN_PASSWORD`). Errors: 400 validation, 403 level gating, 404 session, 409 level done / cap,
429 rate limit, 503 guard or agent unavailable (UI retries once).
Level policy (`configs/levels.yaml`): L1 none, L2 regex, L3 `base` alias, L4 `champion`, L5 `champion` + output filter.
Guard down at L3/L4/L5 fails closed (503), never open; metric `guard_unavailable_total{level,alias,kind}` counts it.
Champion version: the gateway re-reads the MLflow `champion` alias with a 30 s TTL (`gateway/serving.py`); `/health.models.champion` on the guard service should report the same version label.

## 3. Database (Neon Postgres, Alembic revision 0001; models in `breachlab.common.db`)
players, sessions, levels, attempts, labels, model_versions as in PRD 6.1. Level secrets are derived
(`gateway.codes.level_secret`), the DB stores only the salted hash. Gold review rows: session `gold-review-v1`, level 0.

## 4. Configs (keys frozen; values may change)
`configs/train.yaml` (base_model, seed, data_dir, max_len, sft.*, mining.*, dpo.*, calibration.*, budget.*,
hf_namespace) | `configs/serve.yaml` (models, verbalizers.{safe,injection,jailbreak,exfiltration}.{word,form,id},
max_tokens.{input_guard,output_filter}, threshold, temperature) | `configs/gate.yaml` (asr_must_beat_champion,
mcnemar_alpha, fpr_max_delta_pp, fpr_max_abs_pct, p95_latency_ms_max, slice_asr_regression_pp_max, ece_max) |
`configs/levels.yaml` | `configs/eval.yaml` | `configs/budget.yaml`.

## 5. Reports
`reports/gate-input.json`: `{"candidate": {asr, fpr, p95_latency_ms, ece, mcnemar_p, fresh_asr?, slice_asr?},
"champion": {asr, fpr, is_base, fresh_asr?, slice_asr?}}` (fractions in [0,1]).
`web/public/scorecard.json`: `{"version": str, "models": {name: {asr, fpr, hard_benign_fpr, precision, recall, f1,
auroc, category_accuracy, p95_ms, cost_per_1k_usd, note}}}`. README numbers come only from here
(`scripts/readme_numbers.py`). `reports/eval-vN.md` is the human-readable source.

## 6. Registry (MLflow on DagsHub, model `breachlab-guard`)
Aliases: `champion` (served), `challenger` (shadow), `previous` (rollback target). `base` is not a registry alias: it
names the untuned Qwen3-0.6B served for Level 3. Version tags carry `kernel` and `hf_repo`.

## 7. v1.1 game API (Neon Arcade, ADR-0091; additive: every v1 field keeps its meaning)
Bosses: 1 Chatty Bot, 2 Word Cop, 3 The Watcher, 4 Trained Sentinel, 5 Double Lock, 6 Daily Vault (daily only).
Level status: `locked | open | cleared | skipped`. Level N is open when N = 1 or level N-1 is `cleared` or `skipped` (403 otherwise). A skipped level stays playable; clearing it later turns it `cleared` with full points.

7.1 `GET /api/session/{id}` adds `levels: [{id, boss, status, messages_used, points, hints_used: [tiers]}]`, `total_points`, `streak`, `hint_cost` (points per tier).

7.2 `POST /api/message` reply adds
`verdict: {decision: "BLOCKED"|"PASSED", p_attack: float|null, category: "injection"|"jailbreak"|"exfiltration"|"keyword"|"none", layer: "none"|"regex"|"input_guard"|"output_filter"}`
(`p_attack` = guard `p_unsafe` at L3-L6, null at L1-L2), `points_awarded` (0 unless this message cleared the level), `total_points`, `streak`. 202 `{status: "warming", retry_after_s}` unchanged (ADR-0085).

7.3 Scoring (`configs/levels.yaml` key `scoring`): clearing level L gives `max(0, base[L] + per_message_left * messages_left - hint_cost * hints_used)`, times `min(1 + 0.1 * streak, 1.5)` where `streak` counts consecutive clears with no hint and no skip (a hint or a skip resets it to 0). Defaults: base `[100, 200, 400, 800, 1600]`, `per_message_left` 10, `hint_cost` 100. A skip gives 0 points.

7.4 `POST /api/session/{id}/hint {level, tier: 1-3}` -> `{level, tier, text, cost, total_points, hints_used}`. Tiers unlock in order (409 when skipping a tier); re-reading a revealed tier is free; 10/min per session (429). Hint texts live in `configs/levels.yaml` (`hints: [t1, t2, t3]` per level) and never contain a secret. The cost is shown before reveal via `hint_cost`.

7.5 `POST /api/session/{id}/skip {level}` -> `{level, status: "skipped", next_level: int|null, how_others_beat_it: [{prompt, messages}], fallback_hint: str|null, total_points, streak: 0}`. Idempotent (same body on repeat); 409 when the level is cleared; 403 when locked; 10/min per session (429). `how_others_beat_it` holds at most 2 winning prompts for that level from non-canary sessions whose attempt carries a human review label approving display, with that session's secret replaced by `[SECRET]`; when there are none, `fallback_hint` is the tier-3 hint. The response never contains the caller's secret. DB: Alembic revision 0002 adds `levels.skipped_at` (nullable), `levels.points` (int, default 0), `levels.hints_used` (int, default 0).

7.6 `GET /api/leaderboard` entries become `{rank, nickname, beaten, skipped, points, messages, levels: [status x5], guard_version}`, ordered by `beaten` desc, `points` desc, `messages` asc (canary sessions excluded). `hall_of_breaches` unchanged. The UI shows skipped bosses as a grey "SKIP" chip.

7.7 Daily boss: `GET /api/daily` -> `{date (UTC, YYYY-MM-DD), level: 6, boss: "Daily Vault", max_messages: 10, board: [{rank, nickname, messages, at}]}` (top 20, fewest messages then earliest). Play with the normal session and `POST /api/message {level: 6}`; level 6 is always open, uses the champion guard plus the output filter, and its secret is `level_secret(key, "daily:" + date, 6)`, the same for every player that day.

7.8 `GET /api/stats` -> `{players_today, messages_today, bosses_defeated_today, blocks_today, last_breach: {boss, at}|null, updated_at}`; canary sessions excluded; cached 30 s; no nicknames or prompts.

7.9 Share card (web project): `GET https://breach.harshgiri.site/api/og?boss=1-6&msgs=1-20&pts=0-99999&name=<=20 chars` -> 1200x630 PNG (next/og, arcade style, e.g. "I BEAT THE WATCHER IN 4 MESSAGES"); `/share?...` sets `og:image` / `twitter:image` to it. Inputs are validated and clamped; the card never shows a secret.

7.10 Ownership: lane G owns `src/breachlab/gateway/` (except `extras.py`), `configs/levels.yaml`, `migrations/`; lane V owns `src/breachlab/gateway/extras.py` (`/api/daily`, `/api/stats`), `web/app/api/og`, `web/app/share`, `web/app/daily`, `web/components/StatsTicker.tsx`, `web/components/ShareCard.tsx`; lane U owns the rest of `web/`.
