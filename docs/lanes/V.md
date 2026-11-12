# Lane V: daily boss, stats, share card (CONTRACTS.md 7.7-7.9)

Branch `phase-8/v` (local, based on `phase-8/arcade`; not pushed).

## Backend (`src/breachlab/gateway/extras.py`, tests `tests/gateway/test_extras.py`)
- `GET /api/daily` -> `{date, level: 6, boss: "Daily Vault", max_messages: 10, board}`; board = top 20 level-6
  rows cleared on the current UTC date by non-canary sessions, ordered by `messages_used` asc, `cleared_at` asc.
- `GET /api/stats` -> `{players_today, messages_today, bosses_defeated_today, blocks_today, last_breach, updated_at}`.
  players = distinct players with a session started or an attempt today; blocks = attempts with verdict `UNSAFE`
  (regex, input guard or output filter); last_breach = latest non-canary level clear ever (boss name + time).
  Cached 30 s per process (`StatsCache`). No nicknames, prompts or replies.
- Reads only the v1 columns of `breachlab.common.db`; timestamps are naive UTC, returned as ISO with `Z`.
- `build_router(db, settings, *, now=..., cache=...)`: the keyword args exist for tests only.

## Web
- `web/app/api/og/route.tsx` (edge): 1200x630 PNG via `next/og`; font bundled (`PressStart2P-Regular.ttf`, SIL OFL,
  `OFL.txt`), no network needed. `web/app/api/og/card.ts`: input clamping (`parseCard`), boss names, palette
  (`v("pink")` = `var(--pink, #FF3EA5)`), shared by the other files.
- `web/app/share/page.tsx`: og:image / twitter:image = `{origin}/api/og?<clamped params>`, CTA to `/` and `/daily`.
- `web/components/ShareCard.tsx` `{boss, msgs, pts, name}`: navigator.share, else clipboard copy of the `/share` link
  (textarea fallback), plus an X intent link. `web/components/StatsTicker.tsx`: polls `/api/stats` every 30 s, silent
  on failure, static under `prefers-reduced-motion`.
- `web/app/daily/page.tsx` + `DailyGame.tsx`: board, nickname in localStorage `breachlab.nickname`, session id in
  `breachlab.session` (reused; a 404 starts a new session once), `POST /api/message {level: 6}`, verdict chip,
  ShareCard on a clear.
- `web/e2e/share-daily.spec.ts`: 5 mocked tests (run against a server on its own port with `E2E_BASE_URL`, since
  `reuseExistingServer` would otherwise pick up another lane's server on :3100).

## Depends on lane G
- `MessageIn.level` is still `le=5`; the daily game gets 422 until lane G opens level 6 (7.7).
- The daily page uses `points_awarded` / `verdict` from 7.2 when present and degrades without them.
