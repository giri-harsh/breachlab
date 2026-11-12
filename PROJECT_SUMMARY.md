# BreachLab: technical summary

## 1. One-liner
Research-demo web game: players try to make a guarded LLM leak a secret; a fine-tuned Qwen3-0.6B guard (int4, CPU) tries to stop them.

## 2. Problem & why it's hard
Prompt-injection guards must be cheap, low-false-positive and robust to encoded, Hindi and Hinglish attacks. Small models are weak zero-shot (Qwen3-0.6B ASR 100%). Constraints: $0, CPU-only serving, synthetic labels.

## 3. Architecture
`Player → Next.js → FastAPI gateway → [L2 regex | L3-L5 guard service (ONNX)] → vault agent (Groq) → [L5 output filter = same guard] → leak detector → Neon Postgres`
Offline: attempts → judge labels → Kaggle T4 SFT → MLflow `champion` alias, re-read every 30 s.
- One forward pass, softmax over four verbalizer tokens. Beat: generative LLM judge (Gemini ASR 12.8%, FPR 2.8%).
- Fail closed when the guard is down (503/202).
- Deterministic leak detector: free labels.
- int4 weight-only + prefix-KV. Beat: dynamic int8 (FPR 2.3% → 24.3%); fp32 (2384 MB) exceeds the 2 GB function.
- Vercel function host (HF Docker/Gradio Spaces returned 402).
- SFT-only over DPO: no significant gain.

## 4. Stack
Python ≥3.11; FastAPI 0.162.0, Pydantic 2.14.1, SQLAlchemy 2.1.6, httpx 0.28.1; onnxruntime 1.31.1, tokenizers 0.23.3; mlflow-skinny 3.17.2, DVC. Next.js 15.5.28, React 19.1.1, Playwright 1.55.0. Training: transformers ≥4.51, peft, trl (unpinned). Models: Qwen3-0.6B served, 1.7B comparison; Groq fallback chain (gpt-oss-20b first). Infra: Vercel (3 projects), Neon, Kaggle, Hugging Face Hub, DagsHub, GitHub Actions (8 workflows), Langfuse/Grafana via OTLP. No Kubernetes, IaC, queues, Redis or vector store.

## 5. Agentic / LLM specifics
No agent framework, tools, graph, memory or MCP: the vault is one Groq call (max_tokens 400) with a model fallback chain, `retry-after` handling (cap 5 s) and daily-cap parking. Guardrails: regex, input guard, output filter, per-session HMAC secrets, DB-backed caps of 10 msgs/min/session and 30/min/IP. `cost_usd` is never populated, so cost control is the budget config only.
Fine-tune: LoRA r16/α32 on 7 projections, 3 epochs, lr 2e-4, batch 32, fp16 on T4. Data v1: 27,977 rows (train 19,330, val 2,147, gold 1,500, pool 5,000). DPO: 443 pairs.

## 6. Production-readiness
Auth: bearer on admin and guard, HMAC canary header. CI: ruff, mypy strict, pytest ≥80% coverage, eval gate, web build, Docker build. **GitHub Actions is billing-locked, so CI and the hourly canary fail at startup.** Tests (run today): 529 collected, 520 passed, 9 skipped, 98.14% coverage; 17 mocked Playwright plus a live spec. Live (checked 2026-11-01): breachlab-web / -api / -guard `.vercel.app`. Cold-start fix deployed 2026-11-11 (first answer 8.4 s with pre-warm vs 19.65 s before; one run each).

## 7. Evals & metrics
| Metric | Value | Source | n |
|---|---|---|---|
| ASR / FPR, fp32 adapter | 0.6% / 1.6% | `reports/eval-v1.md` | 1500 |
| ASR / FPR / F1 / AUROC, served int4 | 0.59% / 2.08% / 0.985 / 0.999 | same, int4 section | 1500 |
| 3 seeds ASR / FPR | 0.49±0.17% / 1.83±0.24% | same | 1500 |
| Baselines ASR / FPR | DeBERTa 10.4/14.2; Prompt-Guard 51.6/10.3 | same | 1500 |
| Gold-label check | 92.7% agreement, κ 0.851 | same | 1500 |
| Leak suite L1 / L2-L5 | 1/210; 0/216 each | same | 216 |
| Latency c1 (model p50/p95) | 720 / 3468 ms | `docs/lanes/B3-vercel-bench-warm.json` | 500 |
| Latency c4 round trip p50/p95 | 3664 / 9515 ms | same | 500 |
| Throughput c1 / c4 | 0.82 / 0.91 req/s | same | 500 |
| Cold start | 12.8 s after 600 s idle | `B3-vercel-bench-cold.json` | 1 |
| Canary probes | 19/20 | ADR-0084 (manual run) | 20 |
| Short benign opener FPR, fp32 / int4 | 52.0% [46.3,57.7] / 54.3% [48.7,59.7]; Hindi 94-96% | `docs/lanes/S16-opener-fpr.json` | 300 |
| L3 base guard ASR / AUROC | 99.0% / 0.552 | `docs/lanes/S15-base-guard.json` | 1500 |
Gate: `scripts/eval_gate.py reports/gate-input.json` passes (fp32 adapter quality). No chaos or load tests beyond the benchmark; cost per request not measured.

## 8. Scale claims
- Guard p95 ≤300 ms: VERIFIED not met (3468 ms).
- Full-request p95 ≤2 s (PRD): UNVERIFIED, never measured.
- 500 requests at c1/c4: VERIFIED.
- Alias re-read every 30 s: VERIFIED by unit test. "Move about 1.2 s": UNVERIFIED.
- Rate limits 10/30 per minute: PARTIAL (unit tested, not load tested).
- Hourly canary: PARTIAL; one manual run.
- Warm window 10:00-24:00 IST (PRD): UNVERIFIED, not implemented.
- "Spend is $0": PARTIAL; no billing export.
- Cold start no longer 503: PARTIAL; deployed, one run per case, no 503 seen.
- 0.6B keeps 99.6% of 1.7B F1: PARTIAL; fp32, Kaggle 4-vCPU, not served.

## 9. Gaps
No out-of-distribution test (gate skips it); labels not human-reviewed; 300 ms missed; no load or chaos test; no cost tracking; monitoring dormant (billing lock, missing secrets); L3 base guard measured, not deployed; guard blocks 52% of short benign openers; served FPR 2.08% exceeds the 2.0% bar.

## 10. Interview risks
1. *Does it generalize?* Not shown: gold shares sources with train, and fresh short openers are 52% false positives (Hindi 94%).
2. *Why miss 300 ms?* One vCPU; p95 3.5 s.
3. *What does 0/216 mean?* The guard blocked all 216 template attacks; vault and L5 filter untested.
4. *Is the threshold test-tuned?* No, val-derived; yet FPR is 2.08%, and fp32 also blocks Hinglish "rules".
5. *Is it production-ready?* No: one request at a time, no CI, no cost data.
