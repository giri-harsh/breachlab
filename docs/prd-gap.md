# PRD gap check (2026-11-13, FINAL for v1.0.0)

Checked against the public repository `giri-harsh/breachlab` and production (web/api 97f551d, guard v2 `champion-v2`), v1.0.0. Status: DONE, PARTIAL, MISSING, DEFERRED (ADR) or HARSH (needs money, DNS, billing or an irreversible public action). Evidence for every changed row is in the file named.

## Game (PRD 5)
| Item | Status | Evidence |
|---|---|---|
| FR-1 secret per session x level, 8 chars, no ambiguous chars | DONE | `gateway/codes.py` ALPHABET without 0O1lI; HMAC-derived per session and level (equivalent secrecy, nothing stored) |
| FR-2 win on leak | DONE | `service.send` uses `detect()` on the visible reply |
| FR-3 leak detector covers every encoding | DONE | `tests/gateway/test_leak*.py` (base64, hex, ROT13, leet, reversed, separators) |
| FR-4 limits, rate limits, nickname moderation | DONE | `ratelimit.py` (10/min session, 30/min IP), `moderation.check_nickname`, `test_nickname_moderation` |
| FR-5 consent line | DONE | `service.CONSENT`, shown above the input |
| FR-6 scoring | DONE, amended | Harsh's v1.1 rule (bosses beaten, points, fewest messages, skips) in `gateway/service.py`, migration 0002 applied to prod (ADR-0091, ADR-0095), prod e2e |
| FR-7 Hall of Breaches, prompts hidden | DONE | `service.hall_of_breaches` returns no prompt text |
| FR-8 guard waking state | DONE | ADR-0085: 202 "warming"; ADR-0098: `/api/health` guard_state ok/warming/down, BOSS WAKING UP screen, pre-warm, prod cold test (about 22 s to playable), docs/runbook.md |
| FR-9 blocked UX shows category, not confidence | DONE, amended by Harsh | the verdict pill shows "NN% attack" (ADR-0091); prod e2e `arcade.spec` |
| FR-10 canary flag | DONE | `canary_valid` HMAC header, excluded from board and training |
| Pages / leaderboard model-card admin under-the-hood | DONE | `web/app/*` |
| /under-the-hood version timeline | DONE (U) | `web/app/under-the-hood`, e2e `pages.spec` (also shows the guard v2 row) |
| /under-the-hood Grafana public snapshot | DONE-BY-DECISION (ADR-0042, ADR-0107) | OTLP ingest verified; dashboards as JSON in `dashboards/` with an import guide; provisioning needs a Grafana service-account token (Harsh, optional) |

## Guard and eval (PRD 7, 9)
| Item | Status | Evidence |
|---|---|---|
| ASR >= 40% lower than base at FPR <= 2% | DONE | 99.4% relative reduction (`reports/eval-v1.md`) |
| 6 comparison rows, bootstrap CIs, McNemar, slices, 3 seeds | DONE | `reports/eval-v1.md` |
| Size-vs-quality table, 0.6B >= 90% of 1.7B F1 at >= 2x lower latency | DONE (fp32, Kaggle 4-vCPU) | 99.6% at 2.5x |
| Guard p95 <= 300 ms warm on CPU | DONE-BY-DECISION (ADR-0105) | cascade: 136 MB student answers 95% of ordinary traffic (student p50 18-43 ms, p95 27-144 ms), 0.6B only on the uncertain band (4.8% of production requests); held only while escalation < 5%: 9-27% on opener/fresh/OOD traffic (0.6B p95 2,063 ms). `reports/eval-v2.md`, `docs/lanes/D-cascade-v2.json`. Guard-only 500-request prod benchmark needs the guard token (not run) |
| Full request p95 <= 2 s warm | DONE-BY-DECISION (ADR-0105) | prod, sin1, 100 requests c1: gateway time p50 527 / p95 1,142 ms (meets); client round trip from the sandbox p50 2,721 / p95 4,027 ms (before 5,053 / 6,447 ms). `docs/lanes/D-full-request-cascade.json` |
| Cost per 1k requests | DONE (ADR-0107) | $0 actual; $0.1124 per 1,000 requests and $3.37 per 1,000 games at list prices (gpt-oss-20b); `docs/lanes/C-cost.md`, README, model card |
| System leak suite >= 200 attacks per level | DONE | ADR-0083 (216 per level); ADR-0097/0102: 110 fresh creative attacks per boss and the final production difficulty table (`docs/lanes/G-difficulty-prod.json`): boss 3 pooled 21.6%, bosses 4 and 5 0/326, boss 1 fresh 79.1%, boss 2 fresh 57.3% (2.3 points above its band) |
| Guard v2 (hard-benign data) | DONE, champion | ADR-0094: opener FPR 54.3% to 6.3%, gold ASR 0.44%, gold FPR 1.83%; MLflow version 3 is `champion`, v1 is `previous` |
| FPR bar | DONE (ADR-0096) | v1 2.08% within its CI; v2 1.83% meets 2.0% outright |
| ONNX export + CPU serving | DONE (int4; dynamic int8 rejected) | ADR-0057, ADR-0082 |
| Adapters and models on HF | DONE (PUBLIC 2026-11-13) | `Pseudo-Harsh/breachlab-guard-v1` (Apache-2.0), dataset `Pseudo-Harsh/breachlab-data` (CC-BY-SA-3.0), cards link to the GitHub repo (`scripts/hf_publish.py`) |

## MLOps and observability (PRD 10, 11)
| Item | Status | Evidence |
|---|---|---|
| MLflow aliases, promote/rollback workflows, shadow hook, review queue, retraining doc | DONE | `flywheel/registry.py`, `.github/workflows/{promote,rollback}.yml`, `docs/retraining.md` |
| Shadow mode meaningful in prod | DONE-BY-DECISION (ADR-0082, ADR-0105) | challenger and champion share one Vercel host, so the duplicate shadow call was removed (it only added latency); shadow comparison runs offline on Kaggle (ADR-0094) |
| Rollback tested once, timing recorded | DONE (P) | promote median 2.99 s, rollback 4.88 s (`docs/lanes/P-rollback-timing.json`); runbook corrected |
| Gate on every PR | DONE-BY-DECISION (ADR-0104, ADR-0107) | gate = local gate (ADR-0092) until Actions runs; root cause is an account-level billing flag, evidence `docs/lanes/C-ci-evidence.md`, click path in `docs/blockers.md` (Harsh) |
| Langfuse full trace | DONE | progress.md 2026-10-30: 6 observations for one message |
| Grafana dashboards as code / provisioned | DONE-BY-DECISION (ADR-0042) | JSON in `dashboards/` + manual import guide; no API token to provision |
| Alerts to Discord | DONE-BY-DECISION (ADR-0025) | Harsh ruled out Discord/phone alerts; Grafana alert rules exist as JSON |
| Canaries every 15 min | DONE-BY-DECISION (ADR-0041, ADR-0104) | hourly workflow gated on the billing flag; 19/20 by hand (ADR-0084) |
| Weekly drift job | DONE-BY-DECISION (ADR-0104) | workflow exists, gated on the billing flag |

## Non-functional (PRD 12)
| Item | Status | Evidence |
|---|---|---|
| CORS locked to the site | DONE | prod `CORS_ORIGIN` set; a foreign-origin preflight gets 400 (checked 2026-11-12) |
| Admin behind password + rate limit | DONE (P) | password with `compare_digest`; 5 requests/min on `/api/admin/*` (`ratelimit.py`, test) |
| Salted IP hashes, secrets never logged, prompts hidden | DONE | `config.salt`, telemetry redaction |
| Licences stated (code Apache-2.0, dataset CC-BY-SA-3.0) | DONE | `LICENSE`, `DATA_LICENSE.md`, cards (ADR-0100) |
| Public repository | DONE | https://github.com/giri-harsh/breachlab, clean single-commit snapshot, `scripts/prepublic_scan.py` SAFE (ADR-0100) |
| Reproducibility (git SHA + data version + config hash + seed per model) | DONE (P) | `train/repro.py` records git SHA, data hash, config hash, seed in every results.json (v2 run: `data-v1` tag value is stale in the v2 kernel, see ADR-0094) |
| Accessibility: mobile-first, keyboard, WCAG AA | DONE (U) | prod Lighthouse accessibility 100 on /, /play, /leaderboard; performance 98 / 99 / 89 (`docs/lanes/D6-lighthouse-prod.json`); contrast fixes |

## Phase acceptance (PRD 13)
| Phase | Status | Evidence / gap |
|---|---|---|
| 1 Foundation | DONE-BY-DECISION (ADR-0037, ADR-0104) | CI cannot run (account billing flag, Harsh); Modal replaced by Kaggle T4 at $0 |
| 2 Data v1 | DONE | `data-v1` tag at 6e24246, 19,330 train rows, leakage test, dataset card, license table |
| 3 Guard v1 + eval | DONE-BY-DECISION (ADR-0105) | eval-v1 and eval-v2 complete incl. OOD and cascade; 300 ms bar met only on ordinary traffic |
| 4 Game live | DONE-BY-DECISION (ADR-0107) | live and verified at breachlab-web.vercel.app; custom subdomain waits for Harsh's Namecheap CNAME (`docs/blockers.md`) |
| 5 Observability | DONE-BY-DECISION (ADR-0042, ADR-0025) | trace DONE; Discord alerts ruled out by Harsh |
| 6 Registry + gate | DONE-BY-DECISION (ADR-0104) | gate passes locally; Actions cannot run (account flag); rollback timing recorded; v2 promoted through `registry.promote` |
| 7 Polish | DONE | README showcase (demo GIF, screenshots, results, limitations), `check_readme_numbers.py` passes, `scripts/check_links.py` 21 relative and 23 external links OK, 0 broken; LICENSE, DATA_LICENSE, LOCAL_DEV, RESUME_BULLETS, release notes |

## Summary
Every row is DONE or DONE-BY-DECISION (ADR cited). Open items that only Harsh can do: (1) fix the GitHub account billing flag and flip `ACTIONS_BILLING_OK` (ADR-0104, `docs/blockers.md`), then retire the local gate once Actions is green; (2) add the Namecheap CNAME `breachlab` for `breachlab.harshgiri.site`, then set `CORS_ORIGIN` and run `scripts/verify_domain.py`; (3) create the v1.0.0 tag and release. Decided and measured, not met in full: the 300 ms guard p95 and the 2 s full-request bar (cascade, ADR-0105; met on ordinary traffic and server time, not on escalated traffic or the sandbox round trip); student robustness on new attacks (OOD ASR 9.5% cascade vs 6.6% for the 0.6B); gold labels reviewed by LLMs, not humans (ADR-0106). Acceptance: 100% of rows closed, of which the DONE-BY-DECISION rows are listed above with their evidence.
