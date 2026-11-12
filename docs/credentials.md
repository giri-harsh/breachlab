# Credentials: where to get each key

Set all as environment variables in the Claude Code cloud environment (never commit). Then run `scripts/finish_phase1.sh`. GitHub Actions secrets and branch protection: run `scripts/set_gh_secrets.sh` where `gh` is logged in (the session's permission classifier refuses secret-store writes, ADR-0022). Modal GPUs also need a payment method on the Modal workspace (ADR-0020). Accept the `allenai/wildjailbreak` terms on HF if you want it back (ADR-0019).

| Variable(s) | Steps |
|---|---|
| MODAL_TOKEN_ID, MODAL_TOKEN_SECRET | modal.com > Settings > API Tokens > New token |
| HF_TOKEN | huggingface.co/settings/tokens > New token, type Write. Also accept the allenai/wildjailbreak terms on its dataset page |
| GROQ_API_KEY | console.groq.com/keys > Create API Key |
| GEMINI_API_KEY | aistudio.google.com/apikey > Create API key |
| DAGSHUB_TOKEN | dagshub.com > create repo `breachlab` (connect/mirror from GitHub) > User Settings > Tokens. Optional DAGSHUB_USER (default giri-harsh) |
| MLFLOW_TRACKING_URI, _USERNAME, _PASSWORD | DagsHub repo > Remote > Experiments: URI is `https://dagshub.com/<user>/breachlab.mlflow`; username = DagsHub user; password = DAGSHUB_TOKEN |
| NEON_DATABASE_URL | neon.tech > project > Connection Details > pooled connection string |
| LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST (alias LANGFUSE_BASE_URL) | cloud.langfuse.com > project > Settings > API Keys; host `https://cloud.langfuse.com` |
| GRAFANA_OTLP_ENDPOINT, GRAFANA_OTLP_INSTANCE_ID, GRAFANA_OTLP_TOKEN | grafana.com > stack > OpenTelemetry > Configure: OTLP/HTTP endpoint, instance ID (basic-auth user) and generated token (password) |
| VERCEL_TOKEN | vercel.com/account/tokens |
| KAGGLE_API_TOKEN | kaggle.com > Settings > API > Create New Token (single-token format; no username/key pair) |
| ADMIN_PASSWORD | choose any strong password |
| GH_TOKEN | github.com/settings/tokens: classic with `repo` + `workflow`, or fine-grained for the repo with Contents, Pull requests, Actions, Secrets, Administration (branch protection) |
