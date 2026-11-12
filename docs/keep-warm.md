# Guard keep-warm: what is measured, what the canary can do, what it cannot (lane E3)

Relates to ADR-0041 (hourly canary), ADR-0050 (guard failures fail closed), ADR-0082 (guard on a Vercel function).

## Measured facts

All from `docs/lanes/B3-vercel-bench-cold.json` (the real function, https://breachlab-guard.vercel.app, 2026-11-01 07:10-07:23 UTC), as summarised in ADR-0082.

| idle gap before the request | what answered | round trip | notes |
|---|---|---|---|
| 120 s | the same instance (age 1255 s, 627 requests served) | 1061 ms | model time 653 ms |
| 600 s | a new instance (age 10.7 s, 2 requests served) | 12 837 ms | boot 9.7 s (433 MB from the Hub at about 120 MB/s, then the ONNX session load), server wall time 10 396 ms, model time 691 ms |

So an instance stays warm for **at least 2 minutes** and is **gone by 10 minutes**. Where between 2 and 10 minutes it disappears has **not been measured**; in particular nobody has shown that a 5-minute idle gap is survived.

A request that lands on a gone instance boots it inside that request (every route, `/health` included, see `vercel-guard/api/index.py`), which takes about 13 s. The gateway waits `GUARD_TIMEOUT_S=8`, so that player gets the fail-closed 503 "guard waking up" (ADR-0050); the boot carries on and the next request is served by the booted instance. This is a known limit, not a bug.

## What the hourly canary can and cannot do

The canary runs at :17 every hour (`.github/workflows/canary.yml`).

It can:
- tell a cold start from an outage. `monitor/guard_probe.py` calls `GUARD_URL/health` (no auth) with a 25 s timeout. If the first call times out or fails and the retry (5 s later) answers, the run stays green with a warning ("guard cold-start timeout"). Two failures in a row are red (when `GUARD_URL` is set a guard is expected).
- record `guard_ok`, `guard_cold_start_seconds` / `guard_warm_latency_seconds` (first and second `/health` call) and `guard_boot_seconds`. The boot seconds come from the guard itself: `/vercel-info` `boot_s` if it ever carries one, otherwise the `x-guard-boot-s` response header (`/vercel-info` has `instance_age_s`, `served`, `region`, `fetch`, `boot_error` today, not the boot time). `x-guard-cold: 1` marks the response that ran on a fresh instance.
- leave the instance warm for a few minutes after :17, as a side effect of probing it.

It cannot:
- keep the guard warm. Sixty minutes between runs is far beyond the 10 minutes after which the instance is gone, so almost every hourly run finds a cold guard and pays the boot itself. Players who arrive between runs hit the cold start.
- prove anything until the secrets exist (below).

## The one cheap option: a probe every 5 minutes (not added)

A `*/5 * * * *` cron that only does `GET GUARD_URL/health` would be the cheapest in-repo way to keep an instance alive. It is **not added to `canary.yml`**, for three reasons:

1. **Actions minutes.** The repo is private (docs/blockers.md), so minutes are metered. 288 runs a day is about 8 640 runs a month, and each billed at least one minute even when it only skips (docs/runbook.md: "a skipped run bills about one minute"). GitHub's published free allowance for private repos on the Free plan is 2 000 minutes a month (figure from GitHub's pricing page, not verified from this sandbox, and the plan itself is not known here). That would use the allowance in about a week, starve CI, and the budget is $0. The hourly canary alone is about 720 minutes a month.
2. **It is best effort.** GitHub documents scheduled workflows as possibly delayed or dropped under load, so a 5-minute cron does not guarantee a probe every 5 minutes, and a 5-minute gap has not been shown to stay warm (see above).
3. **It cannot run yet.** Actions secrets are still missing (a Harsh blocker, `scripts/set_gh_secrets.sh`). Any guarded workflow would skip green, which is not evidence of a live probe.

If Harsh later makes the repo public (public repos have no minute metering) or accepts the cost, add it guarded exactly like the existing cron (a second `schedule` entry, the same `gate` step that skips when secrets are missing, plus a check that `GUARD_URL` is set; give the keep-warm job its own `concurrency` group so a queued 5-minute run cannot displace a pending hourly canary, and make the 20-probe canary job conditional on `github.event.schedule` so the 5-minute run does not call the vault agent). Before relying on it, measure whether a 5-minute idle gap survives (one `/vercel-info` call before and one after 300 s idle; `instance_age_s` and `served` show whether it is the same instance).

Options that need an account Harsh would have to create, listed without having been tried: a free external uptime monitor pointed at `GUARD_URL/health` (`/health` needs no token), or having the gateway fire a fire-and-forget `/health` at session start so the guard boots while the player is still typing (a gateway change, lane C).

## Update 2026-11-01: pre-warm on intent instead of a cron (ADR-0085)

The decision is still no keep-warm cron. The planned remedy for cold starts is on the player's path, not on a schedule: the web app calls `GET /api/warm` when the page loads and when a session starts, the gateway pings the guard's `/health`, and a first message that still meets a booting guard gets HTTP 202 "warming" and is retried by the UI. This is merged but not yet deployed (Vercel deploy quota, ADR-0088), so the cold-start numbers above are still what production does today; the "after" measurement is pending.

