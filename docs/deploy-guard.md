# Hosting the guard service

## Current host: Vercel function (ADR-0082)

The guard runs today as the Vercel project `breachlab-guard` (https://breachlab-guard.vercel.app, Hobby plan, $0), source in `vercel-guard/`. It is **not** the container image described in the rest of this page.

| | |
|---|---|
| Runtime | Vercel Python function, 1 vCPU, 2 GB, max duration 300 s |
| Graph | one int4 `q4b32` ONNX file (433 MB, MatMulNBits 4-bit, block 32, prefix-KV), `champion/guard-q4b32.onnx` on the private repo `Pseudo-Harsh/breachlab-guard-v1`, fetched into RAM at cold start with `HF_TOKEN` |
| Threshold | its own, 0.23195 (`champion/calibration-q4b32.json`, chosen on the val split at FPR <= 2%); the fp32 threshold fails parity for this graph (ADR-0082) |
| Contract | [`docs/CONTRACTS.md`](CONTRACTS.md) section 1: `POST /score` (bearer `GUARD_TOKEN`, 401 without), `GET /health`, plus `GET /vercel-info` (instance age, requests served, fetch timings, `boot_error`) |
| Gateway env | `GUARD_URL=https://breachlab-guard.vercel.app`, `GUARD_TOKEN`, `GUARD_TIMEOUT_S=8` |
| Deploy | `VERCEL_TOKEN=... HF_TOKEN=... GUARD_TOKEN=... uv run python scripts/deploy_vercel_guard.py`, then `GUARD_URL=... GUARD_TOKEN=... VERCEL_TOKEN=... uv run python scripts/set_guard_url.py` (steps and token rotation: [`docs/runbook.md`](runbook.md)) |

Measured on this host (ADR-0082, ADR-0083; 500 requests per level from the sandbox): concurrency 1 round trip p50 842 / p95 3556 ms (model time p50 720 / p95 3468 ms), concurrency 4 round trip p50 3664 ms, cold first request about 13 s. **The 300 ms target is not met.**

Limits, stated plainly:
- One graph fits, so aliases `base` and `challenger` fall back to the champion: level 3 uses the same guard as level 4. The untuned base guard is not served.
- The single vCPU serializes requests: concurrent players queue.
- After more than about 2 minutes idle the instance may be gone; the first request then takes about 13 s, exceeds the gateway's 8 s wait and gets the fail-closed `503 guard waking up` (ADR-0050). Nothing keeps it warm ([`docs/keep-warm.md`](keep-warm.md)).
- int4 costs some quality: served gold FPR 2.08% against the 2.0% bar (ADR-0083, open item for the owner).

**Why the options below stay.** They are the way to get lower latency (more vCPUs, an always-on instance, the fp32 or `kv` graph) or a second graph so that level 3 can serve the untuned base guard again. Each needs a paid plan or a card on file, which the $0 budget (ADR-0037) rules out for now. To move to one of them: deploy it, run `scripts/set_guard_url.py` with the new `GUARD_URL`, then re-measure latency, re-run the leak suite and update the gate (runbook).

## Alternative hosts: the container image

The guard (champion / challenger / base, ONNX Runtime on CPU) is also one container image. A free Hugging Face Space is not possible (Docker and Gradio Spaces both answer 402, ADR-0063, ADR-0079), so this page gives the exact steps for four other hosts. Nothing here was run against a live host (no Docker daemon, no cloud account in the dev container); what *was* verified is listed in "What is tested" at the end. Whichever host serves `GUARD_URL`, the gateway fails closed at Levels 3-5 when it is unreachable.

Contract: `docs/CONTRACTS.md` section 1 (`POST /score`, `GET /health`). Latency to expect: ADR-0064 (fp32 with
prefix-KV, 2 vCPU emulation: p50 about 0.44 s, p95 about 2.3 s; the 300 ms gate is not met on any CPU host tried).
These are Kaggle-emulation numbers; the real measurement of the shipped host is in the section above (ADR-0082).

## 0. Common to every host

**Build context.** The image needs the vendored runtime modules, so never build from `space/` alone:

```bash
uv run python -m breachlab.serve.space_build --stage-only build/guard   # no network, no token
docker build -t breachlab-guard build/guard                              # any Docker host
```

**Weights are not in the image.** At boot the service downloads, per alias, one ONNX variant from the private Hub
repo `Pseudo-Harsh/breachlab-guard-v1`. Per alias and variant on disk: `kv` or `fp32` 2.38 GB, `w8` 1.12 GB. The
`kv` files exist for `champion`, `challenger` and `base`; an alias without the requested variant falls back to
`fp32` (logged, shown on `/info` under `variants`).

**Configuration (environment variables).**

| name | recommended | meaning |
|---|---|---|
| `GUARD_MODEL_REPO` | `Pseudo-Harsh/breachlab-guard-v1` | Hub repo with `champion/ challenger/ base/` |
| `GUARD_VARIANT` | `kv` | prefix-KV fp32: same logits as fp32 (max diff 1.5e-05), about 2x faster (ADR-0064) |
| `GUARD_ALIASES` | `champion,base` | the gateway calls `champion` (L4/L5) and `base` (L3); `challenger` is shadow-only. Dropping it saves 2.4 GB of RAM and download |
| `GUARD_THREADS` | unset | defaults to the container's cgroup CPU quota, not the host core count |
| `GUARD_TOKEN` (secret) | random 32+ chars | `POST /score` then needs `Authorization: Bearer <token>`; the gateway sends the same value from its own `GUARD_TOKEN`. `/health` and `/info` stay open |
| `HF_TOKEN` (secret) | read-only token, see below | reads the private model repo |
| `GUARD_BOOT_SYNC` | unset | `1` loads the models before the port opens; only for hosts with a long startup probe |
| `GUARD_CACHE_DIR` | `/tmp/hf-guard` (default) or a volume | where the weights are downloaded |

**Memory.** The ONNX Runtime session holds the whole fp32 graph in RAM: about 2.4 GB per loaded alias, plus the
tokenizer and Python (about 0.5 GB). `champion,base` therefore needs about 5.5 GB resident; all three aliases about
8 GB. On hosts whose filesystem is RAM (Cloud Run) the downloaded files count again, so double it. These are
estimates from file sizes: no weights may be loaded in the dev container, so peak RSS has not been measured. After the
first deploy read the real number from the host's memory metric and `GET /info` (`load_s`, `cold_start_s`).
`GUARD_VARIANT=w8` halves the file size but measured about 2x slower (ADR-0057), so it is a last resort.

**Cold start.** `/health` answers 503 `{"ok": false}` while the models load and `/score` answers 503 (the gateway
fails closed and the UI retries once). Measured on the shared Kaggle CPU: model load 7.9 s with a warm file cache and
274 s when the file had to be read cold (bench4). Add the download (2.4 GB per alias from the Hub, tens of seconds on a
datacenter link). Plan for 1-5 minutes from a stopped instance. `min-instances 1` removes it, but only some free
tiers allow an always-on instance (noted per host). A failed boot (Hub 5xx) is retried 3 times automatically;
`GET /info` shows the error.

### Token for the private model repo (read-only, fine-grained)

`HF_TOKEN` in the dev environment is a **write** token for the `Pseudo-Harsh` account. Never put it on a public host.
Create a separate one, signed in as `Pseudo-Harsh`: huggingface.co > Settings > Access Tokens > Create new token >
**Fine-grained** > name `breachlab-guard-read` > under *Repositories* tick "Read access to contents of all selected
repos" and select only `Pseudo-Harsh/breachlab-guard-v1`; leave every other permission unticked. Check it:

```bash
export HF_READ_TOKEN=hf_...       # the new token
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $HF_READ_TOKEN" \
  https://huggingface.co/Pseudo-Harsh/breachlab-guard-v1/resolve/champion/calibration.json   # 200
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $HF_READ_TOKEN" \
  https://huggingface.co/api/whoami-v2                                                      # token valid
```

Generate the service token once and keep it in the same secret stores as the other keys:
`export GUARD_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')`.

## (a) Hugging Face Docker Space (needs PRO, about $9/month)

The only option that deploys with one command from this repo. The deploy token (write) stays on your machine; the
Space receives the read-only token and the service token as secrets.

```bash
export HF_TOKEN=...            # write token of Pseudo-Harsh, used only to create/upload the Space
export HF_READ_TOKEN=...       # fine-grained read-only token (above)
export GUARD_TOKEN=...
uv run python -m breachlab.serve.space_build --space Pseudo-Harsh/breachlab-guard --variant kv \
  --var GUARD_ALIASES=champion,base --secret HF_TOKEN=HF_READ_TOKEN --secret GUARD_TOKEN
# prints https://pseudo-harsh-breachlab-guard.hf.space once the Space is RUNNING
```

Make the Space public (code only, no weights; the default of the command): a private Space needs an HF token as the
bearer, which would collide with `GUARD_TOKEN`. cpu-basic has 2 vCPU and 16 GB RAM, enough for all three aliases
(drop `--var GUARD_ALIASES=...` to load them). The free hardware sleeps after about 48 h without traffic and has an
ephemeral disk, so every wake-up repeats the download; the hourly keep-warm workflow (ADR-0041) prevents the sleep.

## (b) Google Cloud Run (free tier, card on file)

Free tier (verify on the pricing page; figures for us regions): 180,000 vCPU-s, 360,000 GiB-s and 2 M requests per
month with request-based billing, 240,000 vCPU-s and 450,000 GiB-s with instance-based billing. Region
`us-central1`. `min-instances 0` bills only while an instance exists, so a few players a week fit; an always-on
instance (`--min-instances 1`) does **not** fit the free tier at this size.

```bash
export PROJECT=<gcp-project> REGION=us-central1
gcloud config set project "$PROJECT"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
printf %s "$HF_READ_TOKEN" | gcloud secrets create guard-hf-token --data-file=-
printf %s "$GUARD_TOKEN"   | gcloud secrets create guard-token    --data-file=-
SA=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com
for s in guard-hf-token guard-token; do
  gcloud secrets add-iam-policy-binding $s --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor
done

uv run python -m breachlab.serve.space_build --stage-only build/guard
gcloud run deploy breachlab-guard --source build/guard --region "$REGION" --port 7860 \
  --cpu 2 --memory "${MEM:-8Gi}" --concurrency 1 --min-instances 0 --max-instances 1 --timeout 300 \
  --no-cpu-throttling --cpu-boost --allow-unauthenticated \
  --set-env-vars GUARD_VARIANT=kv,GUARD_MODEL_REPO=Pseudo-Harsh/breachlab-guard-v1,GUARD_ALIASES=champion,base \
  --set-secrets HF_TOKEN=guard-hf-token:latest,GUARD_TOKEN=guard-token:latest
gcloud run services describe breachlab-guard --region "$REGION" --format='value(status.url)'   # -> GUARD_URL
```

Notes:

- `--memory`: the file system of a Cloud Run instance is memory, so the downloaded weights (2.4 GB per alias) count
  against the limit on top of the loaded session. `4Gi` only works for a single `w8` alias; the default above is
  `8Gi` for `champion,base`. Raise `MEM=12Gi` for three aliases. Each GiB costs free-tier GiB-seconds: at 8 GiB the
  360,000 GiB-s allowance is about 12.5 hours of billed instance time per month, and every cold start bills the
  1-5 minute boot.
- `--no-cpu-throttling` is required: the models load in a background thread after the port opens, and Cloud Run
  otherwise throttles CPU to near zero outside request handling, which stalls the load. Instance-based billing
  applies (the second free-tier row).
- `--concurrency 1` matches the service lock: one scoring at a time per instance. With `--max-instances 1` extra
  players queue (the gateway timeout then fails closed); raise it only after checking the quota.
- Running `gcloud run deploy --source` builds the image in Cloud Build from the Dockerfile in `build/guard`.

## (c) Fly.io (pay as you go, card required)

Fly has no free allowance for new organisations; with `auto_stop_machines` the cost is the running seconds of one
2-vCPU / 8 GB machine plus a 10 GB volume. The volume keeps the downloaded weights across stops, so a wake-up only
pays the model load.

```bash
uv run python -m breachlab.serve.space_build --stage-only build/guard
cd build/guard                                       # fly.toml (below) goes here, next to the Dockerfile
fly auth login
fly launch --no-deploy --copy-config --name breachlab-guard --region iad
fly volumes create guard_cache --size 10 --region iad
fly secrets set HF_TOKEN="$HF_READ_TOKEN" GUARD_TOKEN="$GUARD_TOKEN"
fly deploy --remote-only
fly ssh console -C "chown -R 1000:1000 /cache"      # once: volumes are root-owned, the app runs as uid 1000
fly machine restart                                  # first boot downloads into /cache
```

`fly.toml` (in `build/guard`):

```toml
app = "breachlab-guard"
primary_region = "iad"

[env]
  GUARD_VARIANT = "kv"
  GUARD_MODEL_REPO = "Pseudo-Harsh/breachlab-guard-v1"
  GUARD_ALIASES = "champion,base"
  GUARD_CACHE_DIR = "/cache"
  PORT = "7860"

[mounts]
  source = "guard_cache"
  destination = "/cache"

[http_service]
  internal_port = 7860
  force_https = true
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0        # 1 = always on (billed 24/7)
  [http_service.concurrency]
    type = "requests"
    soft_limit = 1
    hard_limit = 2

[[http_service.checks]]
  method = "GET"
  path = "/health"
  interval = "30s"
  timeout = "5s"
  grace_period = "10m"            # the models take minutes to load

[[vm]]
  size = "shared-cpu-2x"
  memory = "8gb"
```

## (d) Oracle Cloud Always Free VM (Ampere A1, ARM)

The only host that is free with an always-on instance (so no cold start): up to 4 OCPU / 24 GB RAM of
`VM.Standard.A1.Flex` are Always Free (capacity in a region can run out; retry another availability domain). The
`python:3.11-slim` base and the pinned wheels (`onnxruntime`, `tokenizers`, `numpy`, `hf_xet`) have `aarch64`
builds, so the same Dockerfile builds natively. Oracle can reclaim Always Free instances that stay idle for 7 days;
the hourly keep-warm/canary traffic prevents that. ONNX Runtime speed on Ampere is not measured; re-run the latency
bench (`breachlab.serve.space_bench`) on the VM before relying on the ADR-0064 numbers.

1. Create the instance: Compute > Create instance > image Ubuntu 22.04 (aarch64) > shape `VM.Standard.A1.Flex`, 2 OCPU,
   12 GB RAM, 50 GB boot volume, add your SSH key. In the subnet's security list allow ingress TCP 80 and 443 from
   `0.0.0.0/0`.
2. On the VM:

```bash
sudo apt-get update && sudo apt-get install -y docker.io
sudo iptables -I INPUT 6 -p tcp -m multiport --dports 80,443 -j ACCEPT && sudo netfilter-persistent save   # Oracle images block them by default
# from your machine: copy the staged context (uv run python -m breachlab.serve.space_build --stage-only build/guard)
rsync -a build/guard/ ubuntu@<vm-ip>:guard/
# on the VM:
sudo docker build -t breachlab-guard ~/guard
umask 077 && cat > ~/guard.env <<EOF
HF_TOKEN=<read-only token>
GUARD_TOKEN=<service token>
GUARD_VARIANT=kv
GUARD_MODEL_REPO=Pseudo-Harsh/breachlab-guard-v1
GUARD_ALIASES=champion,base
GUARD_CACHE_DIR=/cache
EOF
sudo docker run -d --name guard --restart unless-stopped -p 127.0.0.1:7860:7860 \
  -v guard-cache:/cache --env-file ~/guard.env breachlab-guard
```

3. HTTPS without DNS work: `<ip with dashes>.sslip.io` resolves to the VM and Caddy gets a Let's Encrypt certificate for it.

```bash
printf '%s\n' '203-0-113-7.sslip.io {' '  reverse_proxy 127.0.0.1:7860' '}' > ~/Caddyfile   # use your IP
sudo docker run -d --name caddy --restart unless-stopped --network host \
  -v ~/Caddyfile:/etc/caddy/Caddyfile -v caddy-data:/data caddy:2
```

`GUARD_URL=https://203-0-113-7.sslip.io`. The `.env` file holds the tokens: keep `chmod 600`, never commit it.

## Point the gateway at it

`GUARD_URL` and `GUARD_TOKEN` are runtime env vars of the Vercel project `breachlab-api`. With `VERCEL_TOKEN` and the
other service variables exported (see the header of `scripts/set_vercel_env.sh`), this sets them and redeploys:

```bash
GUARD_URL=https://<host> GUARD_TOKEN="$GUARD_TOKEN" GUARD_TIMEOUT_S=6 scripts/set_vercel_env.sh
```

`GUARD_TIMEOUT_S` defaults to 3.0 s and the measured p95 is 2.3 s on 2 vCPU, so a slow request would fail closed at
the margin; 6 s leaves headroom. (The Vercel host runs with 8 s; its p95 is 3.5 s.) The preferred hand-off is `scripts/set_guard_url.py`. Variables that are not exported are skipped, not cleared. The guard also has to be
awake: a host that sleeps when idle needs a probe of `/health` (what the hourly canary does and does not do: [`docs/keep-warm.md`](keep-warm.md)).

## Verify

```bash
until [ "$(curl -s -o /dev/null -w '%{http_code}' "$GUARD_URL/health")" = 200 ]; do sleep 15; done   # wait for load
curl -s "$GUARD_URL/health"; echo                  # {"ok":true,"models":{"champion":"sft-0.6b-s0","challenger":null,"base":"base"}}
curl -s "$GUARD_URL/info"; echo                    # variants, threads, load_s, cold_start_s, error
GUARD_URL=https://<host> GUARD_TOKEN="$GUARD_TOKEN" \
  uv run pytest tests/gateway/test_guard_contract.py -o addopts="" -q
```

The contract suite runs the in-process fake and the live service (the `live` parameter is skipped when `GUARD_URL` is
unset). With `GUARD_ALIASES=champion,base` the `challenger` entry of `/health` is `null`, which the contract allows;
the alias-fallback test then serves `challenger` requests with the champion. `python -m breachlab.monitor.canary
--ingest-test` is not needed for this check.

Then re-measure on the final host and record it: `uv run python -m breachlab.serve.space_bench --url $GUARD_URL
--gold <gold_test.jsonl> --out latency.json` (500 requests, concurrency 1 and 4; sends `GUARD_TOKEN` as the bearer), and set `configs/gate.yaml` `p95_latency_ms_max` to measured x 1.2 (orchestrator).

## What is tested (and what is not)

- In CI/dev: `tests/serve/test_space_contract.py` runs the shipped FastAPI app (fake ONNX sessions) through the same
  contract assertions as `tests/gateway/test_guard_contract.py`; `tests/serve/test_space_app*.py` cover auth, aliases,
  variant fallback, boot retry and cgroup-aware threads; `tests/serve/test_space_image.py` stages the build context,
  imports `app` from it with the repo off the path (vendored closure complete) and lints the Dockerfile.
- Not tested here: `docker build`, any real deploy, real ONNX inference inside the container image, peak memory and
  the Ampere/ARM speed. Treat the `--memory` and `fly.toml` values as starting points.
