# Lane C, L8: custom domain breachlab.harshgiri.site (Vercel side done, DNS is Harsh's)

Date: 2026-11-13. Only one change was made on Vercel: the domain was added to the web project.

## What was done (Vercel REST API, team `team_yIXRPPOQWbqXkM682GSKqORh`)

1. `POST /v10/projects/prj_mWcWKnDvOQfVL99vf9MHbCIgWR8P/domains` with `{"name":"breachlab.harshgiri.site"}` returned 200:
   `{"name":"breachlab.harshgiri.site","apexName":"harshgiri.site","projectId":"prj_mWcWKnDvOQfVL99vf9MHbCIgWR8P","redirect":null,"verified":true}`.
   The project now lists two domains: `breachlab-web.vercel.app` and `breachlab.harshgiri.site`.
2. `GET /v9/projects/{id}/domains/breachlab.harshgiri.site` -> `verified: true`. No TXT verification record is needed:
   the apex `harshgiri.site` already belongs to this team (`GET /v6/domains/harshgiri.site`: `verified: true`, `serviceType: external`,
   nameservers `dns1.registrar-servers.com` and `dns2.registrar-servers.com`, i.e. the DNS is hosted at Namecheap).
3. `GET /v6/domains/breachlab.harshgiri.site/config` -> `misconfigured: true` (no record yet) and the exact records:
   - `recommendedCNAME` rank 1: `d451dc7baa80fa91.vercel-dns-017.com.` (project-specific target, preferred)
   - `recommendedCNAME` rank 2: `cname.vercel-dns.com.` (generic target, still valid)
   - `recommendedIPv4` (only for apex/A records, not needed for a subdomain): `216.198.79.1`, `64.29.17.1` (rank 1), `76.76.21.21` (rank 2)
   - `conflicts: []`, `cnames: []`, `aValues: []`: nothing at that name today.
   DNS-over-HTTPS confirms `breachlab.harshgiri.site` is NXDOMAIN right now, so adding the CNAME will not collide with an existing record.

The exact record to create, and the steps, are in `docs/blockers.md` ("Subdomain DNS (needs Harsh or his assistant)").

## Checking it

`python scripts/verify_domain.py` (stdlib only; uses dnspython when installed, otherwise DNS over HTTPS, then `nslookup`):
DNS CNAME target, `GET https://breachlab.harshgiri.site/` 200, a CORS preflight from the new origin to the gateway, and
(with `VERCEL_TOKEN`) `verified` and `misconfigured: false`. `--wait 600` polls DNS for up to ten minutes. Exit code 0 only if every check passes.
Today (before the CNAME exists) it prints FAIL for DNS, HTTPS, CORS and Vercel, and INFO for `/api/health`.

Note on `/api/health`: the web project (`breachlab-web`) has no such route (it answers 404). The health endpoint is the gateway's:
`https://breachlab-api.vercel.app/api/health`. The new domain only fronts the web UI, which calls the gateway cross-origin
(`NEXT_PUBLIC_API_BASE=https://breachlab-api.vercel.app` on the web project, plain env, production and preview).

## CORS: what must change when the domain goes live (not changed now)

- `src/breachlab/gateway/config.py`: `cors_origin` is read from `CORS_ORIGIN` (default `"*"`).
- `src/breachlab/gateway/app.py` (CORSMiddleware): `allow_origins=[o.strip() for o in st.cors_origin.split(",")]`, methods `GET, POST`, headers `*`, no credentials.
  So the variable is a comma-separated list of exact origins (scheme + host, no path, no trailing slash); no wildcard subdomains.
- Production value today (set per ADR/`scripts/set_vercel_env.sh`): `https://breachlab-web.vercel.app`. Verified live: a preflight from `https://breachlab.harshgiri.site`
  to `https://breachlab-api.vercel.app/api/message` returns `400 Disallowed CORS origin`, so the new domain would load but every game call from the browser would fail until the variable changes.
- Change to (keep the old origin as long as old links circulate):
  `CORS_ORIGIN=https://breachlab-web.vercel.app,https://breachlab.harshgiri.site`
  on project `breachlab-api` (`prj_xSY0LQ5tzcH3fdGT7EbDXbvk551N`), env id `MNyAU0Z17eqZtkl9` (type encrypted, target production).
  REST: `PATCH /v9/projects/prj_xSY0LQ5tzcH3fdGT7EbDXbvk551N/env/MNyAU0Z17eqZtkl9?teamId=team_yIXRPPOQWbqXkM682GSKqORh` with `{"value":"https://breachlab-web.vercel.app,https://breachlab.harshgiri.site"}`.
  Env changes apply to new deployments only: redeploy `breachlab-api` (the orchestrator batches deploys; ADR-0101 limits 90 per day) and then run `scripts/verify_domain.py`.
- Also update `scripts/set_vercel_env.sh` (default `CORS_ORIGIN`) so a future env re-seed does not drop the new origin.
- Order: add the DNS record, wait for `misconfigured: false` (Vercel issues the certificate automatically), set `CORS_ORIGIN`, redeploy the API, then run the verifier, and only then change links.

## Links to update after it is live

Listed (not changed) in `docs/lanes/C-domain-links.txt`: four code/script files (`web/app/share/page.tsx`, `web/components/ShareCard.tsx`,
`scripts/set_vercel_env.sh`, `scripts/set_guard_url.py`) and the README, runbook, release notes, contracts, demo script, CLAUDE.md and tests.
Regenerate the list with `python scripts/verify_domain.py --write-links docs/lanes/C-domain-links.txt`.
