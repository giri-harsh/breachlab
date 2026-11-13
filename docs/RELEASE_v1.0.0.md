# BreachLab v1.0.0

A web game where you try to trick a guarded AI vault into leaking a secret code across five bosses, built end to end as an MLOps project on free tiers ($0). Research demo, not a production safety system.

## What's live
- **Game:** https://breachlab-web.vercel.app (Neon Arcade UI: 5 bosses + a daily boss, hints, skip boss, points and streaks, leaderboard, share card) with the gateway at https://breachlab-api.vercel.app/api/health.
- **Guard v2** (Qwen3-0.6B + LoRA, SFT, int4 ONNX with prefix-KV reuse) serving bosses 3 to 5 from a 1 vCPU Vercel function; MLflow registry version 3 is `champion`, v1 is `previous` (rollback measured at a median 4.9 s).
- **Hugging Face (public):** [model, Apache-2.0](https://huggingface.co/Pseudo-Harsh/breachlab-guard-v1) and [dataset, CC-BY-SA-3.0](https://huggingface.co/datasets/Pseudo-Harsh/breachlab-data).
- Guard status is `ok | warming | down` at runtime (a cold guard shows "BOSS WAKING UP", never "offline"); the gateway fails closed.

## Headline numbers (gold_test, 1,500 rows; `reports/eval-v1.md`, `reports/eval-v2.md`)
| | attack success (lower is better) | false positives | AUROC |
|---|---|---|---|
| Qwen3-0.6B zero-shot | 100.0% | 0.4% | 0.591 |
| fine-tuned, fp32 adapter (v1) | 0.6% | 1.6% | 0.999 |
| int4 as served, v1 | 0.6% | 2.1% | 0.999 |
| **int4 as served, v2 (this release)** | **0.4%** | **1.8%** | **1.000** (0.9998) |

Short benign openers (300 held-out rows): false positives **54.3% (v1) to 6.3% (v2)**; Hindi 96% to 15.0% (at its 15% gate).

## Game difficulty (production, guard v2, 110 fresh attacks per boss; ADR-0102)
| Boss | Target leak rate | Fresh 110 | 216 template attacks |
|---|---|---|---|
| 1 Chatty Bot | 70%+ | 79.1% | 76.4% (local full run) |
| 2 Word Cop | 40-55% | 57.3% | 51.4% (replay of the full run) |
| 3 The Watcher | 20-35% | 30.0% (21.6% pooled) | 17.3% |
| 4 Trained Sentinel | 10% or less | 0% | 0% |
| 5 Double Lock | 5% or less | 0% | 0% |

## Known limitations
- **Latency misses the 300 ms bar:** model time p50 about 720 ms and p95 2,063 ms warm on the free 1 vCPU host (v1: 3,468 ms); full request p95 6.4 s (v1 guard). **Cold start:** about 13 s for the guard, about 22 s from page load to a playable boss 3 when both functions are cold.
- **In-distribution evaluation:** gold_test shares sources and generators with training; gold labels are generation-time labels (Gemini agreement 92.7%, kappa 0.851), not human-reviewed. The opener set is LLM-written and small.
- **0% on templated attacks is not "unbreakable":** human players will find gaps. Boss 2 and 3 difficulty depends on which free-tier Groq model answers.
- v2 is one seed; no fp32 parity run for the v2 graph.
- GitHub Actions is blocked by a billing lock on the owner's account, so workflows are gated off (ADR-0103/0104) and merges used a documented local gate (ADR-0092).

## Licences
Code, weights and model card: Apache-2.0. Dataset: CC-BY-SA-3.0 (the model's training data includes CC-BY-SA-3.0 rows). See `LICENSE`, `DATA_LICENSE.md`.

Full details: `README.md`, `docs/decisions.md` (ADRs), `docs/prd-gap.md`, `docs/RESUME_BULLETS.md`.
