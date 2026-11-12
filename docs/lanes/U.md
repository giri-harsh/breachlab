# Lane U: Neon Arcade web UI (phase-8/u)

Branch `phase-8/u` from `phase-8/arcade` (local only: no push, no PR, no deploy). Owns `web/` except lane V's files
(`web/app/api/og`, `web/app/share`, `web/app/daily`, `web/components/StatsTicker.tsx`, `web/components/ShareCard.tsx`),
which are imported unchanged (`<StatsTicker />` on `/`, `<ShareCard boss msgs pts name />` in the BOSS DEFEATED sheet).

## What changed
- Tokens in `web/app/globals.css`: `--bg --surface --raised --border --pink --cyan --yellow --green --text --muted --muted-2`
  (plus `--border-ui`, `--soft`, `--ink`, `--ink-cyan`, `--*-dark` shadow shades, `--chip-grey(-bg)`), utility classes
  `font-display` (Press Start 2P via next/font, only for short labels, >= 10px) and `font-body` (Space Grotesk), plus
  `.cabinet`, `.grid-bg`, `.btn-*`, `.field`, `.pill`, `.chip`. Tailwind colours map to the same variables.
- `/`: cabinet hero (INSERT PROMPT TO CONTINUE, BREAK THE BOT, player name, START -> `/play?boss=1`; CONTINUE AS NAME
  when a session is saved), SELECT BOSS grid (status from `GET /api/session/{id}` `levels[]`; the v1 `cleared` field is
  also understood), HIGH SCORES preview (top 5), `<StatsTicker />`.
- `/play`: boss tabs, boss header with SVG hearts + count, score/streak, chat with a verdict pill per reply
  (`BLOCKED|PASSED · NN% attack · category · layer`; the percent is left out when `p_attack` is null; v1 replies fall back to
  `blocked/block_category`), POWER-UPS (6 cards that insert an editable starter prompt and never send), HINT TOKENS (cost
  shown, confirm, then `POST /hint`; revealed tiers are re-read for free), SKIP BOSS (exact confirm text, `POST /skip`,
  HOW OTHERS BEAT IT or the fallback hint, NEXT BOSS; skipped bosses can be replayed), BOSS DEFEATED (the leaked reply
  assembles character by character, then canvas confetti, card + ShareCard + NEXT BOSS), GAME OVER · CONTINUE? (new run
  with the same name, or SKIP), BOSS WAKING UP… (the existing 202 auto-retry in `web/lib/api.ts` is unchanged),
  guard-offline fail-closed and vault-busy messages, consent line. Enter fires, Shift+Enter adds a newline, Esc clears,
  autofocus, FIRE disabled while pending, 500-char counter, input sticky at the bottom on phones, hit targets >= 44px.
- Header HUD: CREDITS ∞, HI score (best `total_points` seen, localStorage), sound toggle (WebAudio blips, default OFF).
- `/leaderboard` HIGH SCORES (API order, green boss chips, grey SKIP chips, v1 rows still render) + Hall of Breaches.
- `/under-the-hood`: animated request path regex -> input guard -> vault -> output filter -> leak detector with numbers
  from `public/scorecard.json`, scorecard table, latency note, VERSION TIMELINE (closes the PRD gap), flywheel.
- `/model-card` and `/admin/review` restyled. `prefers-reduced-motion` turns off all CSS animation, the confetti and the assemble effect.
- Every localStorage access is wrapped in try/catch (`web/lib/game.ts` `store`).

## Tests (mocked API; projects mobile-390 = 390x844 with a Pixel 7 UA, desktop-1440 = 1440x900)
`PW_CHROMIUM=/opt/pw-browsers/chromium npx playwright test`: **64 passed, 19 skipped** (7 live tests without
E2E_LIVE, 12 screenshot tests without SHOTS=1).
- `e2e/arcade.spec.ts`: home -> /play flow, Enter/Shift+Enter/Esc/counter, L1 beaten via a power-up, L2 beaten by
  rephrasing after a regex block, verdict with attack %, hint token (cost first), SKIP -> next unlocked -> replay of the
  skipped boss, fallback hint, GAME OVER, warming, sound + HI, HIGH SCORES SKIP chips. `e2e/mock.ts` is a stateful v1.1 mock.
- `e2e/motion.spec.ts`: reduced motion. `test.use({ reducedMotion })` is not applied with the container Chromium, so the spec uses `page.emulateMedia`.
- Ported: `offline-ux`, `cold-start`, `pages`, `play`. `live.spec.ts` now uses `/play?unlock=1`, `boss-tab-N`, FIRE,
  the `verdict` pill and the `boss-defeated` sheet. It runs on mobile-390 only. It was not run here (no live runs in this lane).
- `npm run lint`, `npm run typecheck`, `npm run build`: pass.

## Lighthouse (mobile, `next start`, lighthouse 12.8.2, CHROME_PATH=/opt/pw-browsers/chromium, headless)
| page | performance | accessibility | best practices | SEO | LCP | CLS |
|---|---|---|---|---|---|---|
| / | 97 | 100 | 96 | 100 | 2.2 s | 0.067 |
| /play | 96 | 100 | 96 | 100 | 2.2 s | 0.092 |
| /leaderboard | 98 | 100 | 96 | 100 | 2.0 s | 0 |
| /under-the-hood | 99 | 100 | 96 | 100 | 1.6 s | 0.001 |
| /model-card | 99 | 100 | 96 | 100 | 1.9 s | 0 |
| /admin/review | 99 | 100 | 96 | 100 | 2.0 s | 0 |

Best practices is 96 on every page because of `errors-in-console`: `/api/health` and `/api/leaderboard` return 404 on
localhost (no gateway behind `next start`). /play first measured 89 (CLS 0.179, from the LOADING placeholder being
swapped for the form). The loading shell is now the same cabinet (`components/InsertName.tsx`).

## Contrast (WCAG AA), `node web/scripts/contrast.mjs` (exits 1 if a pair in use fails)
`--muted-2` is lightened from the design's #8A74C4 (3.80:1 on raised, fails) to #A08BDB. Form-control borders use
`--border-ui` #7A5BC7 (>= 3:1). `--border` #4A2A8A stays for decorative card outlines only.

| foreground | background | ratio | AA needs | result |
|---|---|---|---|---|
| text #F2EAFF | bg #12062B | 16.55 | 4.5 (text) | PASS |
| text #F2EAFF | surface #1E0B45 | 15.13 | 4.5 (text) | PASS |
| text #F2EAFF | raised #2E1566 | 12.71 | 4.5 (text) | PASS |
| muted #B9A6E8 | bg #12062B | 8.91 | 4.5 (text) | PASS |
| muted #B9A6E8 | surface #1E0B45 | 8.14 | 4.5 (text) | PASS |
| muted #B9A6E8 | raised #2E1566 | 6.84 | 4.5 (text) | PASS |
| muted-2 #A08BDB | bg #12062B | 6.64 | 4.5 (text) | PASS |
| muted-2 #A08BDB | surface #1E0B45 | 6.06 | 4.5 (text) | PASS |
| muted-2 #A08BDB | raised #2E1566 | 5.09 | 4.5 (text) | PASS |
| soft #D7C8FF | bg #12062B | 12.51 | 4.5 (text) | PASS |
| soft #D7C8FF | surface #1E0B45 | 11.43 | 4.5 (text) | PASS |
| soft #D7C8FF | raised #2E1566 | 9.61 | 4.5 (text) | PASS |
| pink #FF3EA5 | bg #12062B | 5.97 | 4.5 (text) | PASS |
| pink #FF3EA5 | surface #1E0B45 | 5.45 | 4.5 (text) | PASS |
| pink #FF3EA5 | raised #2E1566 | 4.58 | 4.5 (text) | PASS |
| cyan #3EF0FF | bg #12062B | 13.91 | 4.5 (text) | PASS |
| cyan #3EF0FF | surface #1E0B45 | 12.71 | 4.5 (text) | PASS |
| cyan #3EF0FF | raised #2E1566 | 10.68 | 4.5 (text) | PASS |
| yellow #FFE14D | bg #12062B | 14.84 | 4.5 (text) | PASS |
| yellow #FFE14D | surface #1E0B45 | 13.57 | 4.5 (text) | PASS |
| yellow #FFE14D | raised #2E1566 | 11.40 | 4.5 (text) | PASS |
| green #5BFF8A | bg #12062B | 14.83 | 4.5 (text) | PASS |
| green #5BFF8A | surface #1E0B45 | 13.55 | 4.5 (text) | PASS |
| green #5BFF8A | raised #2E1566 | 11.39 | 4.5 (text) | PASS |
| muted-2-orig #8A74C4 | bg #12062B | 4.95 | 4.5 (text) | PASS |
| muted-2-orig #8A74C4 | surface #1E0B45 | 4.52 | 4.5 (text) | PASS |
| muted-2-orig #8A74C4 | raised #2E1566 | 3.80 | 4.5 (text) | FAIL (replaced, not used for text) |
| ink #2A0A3F | yellow #FFE14D | 13.29 | 4.5 (text) | PASS |
| ink #2A0A3F | pink #FF3EA5 | 5.34 | 4.5 (text) | PASS |
| ink #2A0A3F | green #5BFF8A | 13.28 | 4.5 (text) | PASS |
| ink #2A0A3F | cyan #3EF0FF | 12.45 | 4.5 (text) | PASS |
| ink-cyan #10213A | cyan #3EF0FF | 11.62 | 4.5 (text) | PASS |
| chip-grey #C9C2D9 | chip-grey-bg #3A3150 | 7.03 | 4.5 (text) | PASS |
| border-ui #7A5BC7 | bg #12062B | 3.82 | 3 (UI border/icon) | PASS |
| border-ui #7A5BC7 | surface #1E0B45 | 3.49 | 3 (UI border/icon) | PASS |
| border-orig #4A2A8A | bg #12062B | 1.83 | 3 (decorative only) | FAIL (replaced, not used for text) |
| pink #FF3EA5 | bg #12062B | 5.97 | 3 (UI border/icon) | PASS |
| cyan #3EF0FF | bg #12062B | 13.91 | 3 (UI border/icon) | PASS |
| yellow #FFE14D | bg #12062B | 14.84 | 3 (UI border/icon) | PASS |

failing pairs in use: 0

## Screenshots (`SHOTS=1 npx playwright test screens`, in docs/screenshots/arcade/, each at 390 and 1440)
home, play-start, play (mid-boss: verdicts, a hint, a power-up loaded), leaderboard, under-the-hood, model-card, admin,
moment-boss-defeated (with confetti), moment-game-over, moment-skip-confirm, moment-skip-how-others, moment-warming.

## Open items
- CONTINUE after GAME OVER starts a new session (same name, back to boss 1). The API caps messages per session and level,
  so there is no way to retry in place. A per-level reset endpoint would need lane G.
- The real gateway stays v1 until lane G lands v1.1. Until then hint and skip return 404 there and the UI shows the error
  text; verdict, points and HI fall back to the v1 fields. `live.spec.ts` was updated but not run.
- ShareCard and StatsTicker are still lane V placeholders that render nothing. The win sheet wraps ShareCard in `share-card-slot`.
- The guard-offline banner and 503 copy keep the v1 wording ("Levels 3-5", "Levels 1 and 2") because the specs assert it.
